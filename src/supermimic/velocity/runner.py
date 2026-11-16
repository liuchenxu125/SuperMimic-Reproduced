"""Independent native Trainer subclass. No monkeypatches or global cfg changes."""

import hashlib
import json
import os
import random
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path

import numpy as np
import torch

from supermimic.config import TrainConfig
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.robots import get_robot
from supermimic.training.loop import Trainer

from .commands import VelocityCommands
from .policy import VelocityEncoder, mixed_actions
from .training import VelocityPolicyTrainer, motion_count

FORMAT = "supermimic-motion-velocity-v1"


def restore_config(values):
    def fill(obj, values):
        known = {f.name for f in fields(obj)}
        for key, value in values.items():
            if key not in known:
                raise ValueError(f"Unknown native configuration field: {key}")
            current = getattr(obj, key)
            if is_dataclass(current):
                fill(current, value)
            else:
                if isinstance(current, Path) and value is not None:
                    value = Path(value)
                elif isinstance(current, tuple):
                    value = tuple(value)
                setattr(obj, key, value)
        return obj

    return fill(TrainConfig(), values)


def dataset_contract(cfg):
    files = (
        [cfg.data.library]
        if cfg.data.library
        else discover_npz(
            cfg.data.npz_dir, limit=cfg.data.limit, names=cfg.data.clips, manifest=cfg.data.manifest
        )
    )
    if not files:
        raise FileNotFoundError(f"No training clips under {cfg.data.npz_dir}")
    entries = [
        (str(Path(p).resolve()), Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in files
    ]
    manifest_sha = (
        hashlib.sha256(Path(cfg.data.manifest).read_bytes()).hexdigest()
        if cfg.data.manifest else None
    )
    payload = json.dumps([entries, manifest_sha], ensure_ascii=False).encode()
    return {
        "file_count": len(files),
        "file_metadata_manifest_sha256": hashlib.sha256(payload).hexdigest(),
    }


def robot_contract(robot):
    import mujoco

    profile = get_robot(robot)
    model = profile.model(collision_model="primitive" if robot == "casbot02" else "mesh")
    params = profile.motor_parameters()
    binary = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=binary)
    return {
        "robot": robot,
        "joint_names": [model.joint(i).name.removeprefix("robot/") for i in range(1, model.njnt)],
        "motors": params,
        "compiled_model_sha256": hashlib.sha256(binary.tobytes()).hexdigest(),
    }


class VelocityTrainer(Trainer):
    def __init__(self, cfg, velocity_cfg, init_checkpoint=None):
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("Independent velocity task currently supports one GPU/process")
        if cfg.robot != "casbot02":
            raise ValueError("This independent task currently targets Casbot02")
        velocity_cfg.validate()
        if cfg.env.anchor_obs:
            raise ValueError("Velocity joint task preserves no-root-position motion observations")
        if cfg.policy_trainer.batch_size < 2 or cfg.env.num_envs < 2:
            raise ValueError("Need at least two simulation slots and two imagination slots")
        self._velocity_ready = False
        self._pending_resume = None
        self.velocity_cfg = velocity_cfg
        self.data_identity, self.robot_identity = dataset_contract(cfg), robot_contract(cfg.robot)
        if cfg.resume:
            saved = torch.load(cfg.resume, map_location="cpu", weights_only=False)
            if saved.get("velocity_format") != FORMAT:
                raise ValueError(
                    "--resume requires a motion+velocity checkpoint; use --init-checkpoint for motion weights"
                )
            if saved["velocity_config"] != asdict(velocity_cfg):
                raise ValueError("Resume velocity task settings changed")
            if (
                saved["data_identity"] != self.data_identity
                or saved["robot_identity"] != self.robot_identity
            ):
                raise ValueError("Resume data or robot motor contract changed")
        super().__init__(cfg)
        if self.world != 1:
            raise ValueError("Independent velocity task is single-GPU")
        self.policy.encoders["velocity"] = VelocityEncoder(cfg.policy).to(self.device)
        self.initialization = {"mode": "scratch"}
        if init_checkpoint is not None:
            old = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
            source = old["config"]
            if (
                source.get("robot", "g1") != cfg.robot
                or source["policy"]["goal_dim"] != cfg.policy.goal_dim
                or tuple(source["policy"]["obs_term_scale"]) != tuple(cfg.policy.obs_term_scale)
            ):
                raise ValueError("Initialization motion observation/goal layout differs")
            result = self.policy.load_state_dict(old["policy"], strict=False)
            if result.unexpected_keys or any(
                not k.startswith("encoders.velocity.") for k in result.missing_keys
            ):
                raise ValueError("Initialization policy has unexpected interfaces/weights")
            self.wm.load_state_dict(old["wm"])
            self.critic.load_state_dict(old["critic"])
            self.initialization = {
                "mode": "weights-only",
                "checkpoint": str(Path(init_checkpoint).resolve()),
                "source_iteration": old["iteration"],
                "sha256": hashlib.sha256(Path(init_checkpoint).read_bytes()).hexdigest(),
            }
        self.velocity_commands = VelocityCommands(self.lib, velocity_cfg)
        self.policy_trainer = VelocityPolicyTrainer(
            self.wm,
            self.policy,
            self.critic,
            self.lib,
            cfg.policy_trainer,
            self.velocity_commands,
            velocity_cfg,
        )
        self.motion_envs = motion_count(cfg.env.num_envs, velocity_cfg.velocity_fraction)
        self.collected = {"motion": 0, "velocity": 0}
        self._velocity_ready = True
        if self._pending_resume is not None:
            self.load(cfg.resume, checkpoint=self._pending_resume)
            self._pending_resume = None
        metadata = {
            "format": FORMAT,
            "velocity_config": asdict(velocity_cfg),
            "data_identity": self.data_identity,
            "robot_identity": self.robot_identity,
            "initialization": self.initialization,
            "motion_envs": self.motion_envs,
            "velocity_envs": cfg.env.num_envs - self.motion_envs,
            "motion_history_dim": cfg.policy.history
            * (3 * cfg.policy.num_joints + (12 if cfg.env.anchor_ori_obs else 6)),
            "velocity_proprio_dim": cfg.policy.history * (3 * cfg.policy.num_joints + 6),
            "motion_goal_dim": cfg.policy.goal_dim,
            "velocity_goal_dim": 3,
            "actor_parameters": sum(p.numel() for p in self.policy.parameters()),
            "velocity_encoder_parameters": sum(
                p.numel() for p in self.policy.encoders["velocity"].parameters()
            ),
            "reward": "native motion tracking for both interfaces",
            "smoothing_boundary": "per-clip",
            "velocity_orientation_padding": "constant identity; no motion-reference observation",
            "cuda_cache_handoff": "release_unused_torch_cache_after_policy_update",
        }
        (self.log_dir / "velocity_config.json").write_text(
            json.dumps(metadata, indent=2, default=str)
        )
        print(
            "[VELOCITY] "
            + json.dumps(
                {
                    k: metadata[k]
                    for k in (
                        "motion_envs",
                        "velocity_envs",
                        "motion_history_dim",
                        "velocity_proprio_dim",
                        "motion_goal_dim",
                        "velocity_goal_dim",
                        "actor_parameters",
                        "velocity_encoder_parameters",
                        "reward",
                        "initialization",
                        "cuda_cache_handoff",
                    )
                }
            ),
            flush=True,
        )

    @torch.no_grad()
    def act(self, obs):
        command = self.velocity_commands(self.command.frame, self.command.robot_root_quat_w)
        self.policy.encoders["velocity"].normalizer.update(command[self.motion_envs :])
        return mixed_actions(self.policy, obs["policy"], obs["goal"], command, self.motion_envs)

    def _collect_step(self):
        super()._collect_step()
        self.collected["motion"] += self.motion_envs
        self.collected["velocity"] += self.cfg.env.num_envs - self.motion_envs

    def _synchronize(self):
        super()._synchronize()
        if self._velocity_ready and self.policy_trainer.latest:
            if (self.iteration + 1) % self.cfg.schedule.log_interval == 0 or self.iteration == 0:
                record = {
                    "iteration": self.iteration + 1,
                    "collected": self.collected,
                    **self.policy_trainer.latest,
                }
                with (self.log_dir / "velocity_metrics.jsonl").open("a") as f:
                    f.write(json.dumps(record) + "\n")
                print("[VELOCITY] " + json.dumps(record), flush=True)
            self.policy_trainer.latest = {}
            if str(self.device).startswith("cuda"):
                # Backprop has returned and native synchronization has completed.
                # Warp cannot reclaim Torch's idle cache when launching its next
                # physics graph. Return those blocks once per update, retaining
                # all live model, optimizer and replay tensors.
                torch.cuda.empty_cache()

    def save(self, name="ckpt_latest.pt"):
        ck = {
            "checkpoint_version": 2,
            "iteration_unit": "update",
            "iteration": self.iteration,
            "num_updates": self.num_updates,
            "env_steps": self.env_steps,
            "replay_refill_env_steps": self.replay_refill_env_steps,
            "collection_accumulator": self._collection_acc,
            "train_seconds": self._train_seconds,
            "precision": self.precision,
            "resume_migration": self.resume_migration,
            "wm": self.wm.state_dict(),
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "wm_opt": self.wm_trainer.opt.state_dict(),
            "policy_opt": self.policy_trainer.opt.state_dict(),
            "critic_opt": self.policy_trainer.critic_opt.state_dict(),
            "bin_return": self.command.bin_return,
            "bin_memory": self.command.bin_memory,
            "config": asdict(self.cfg),
            "velocity_format": FORMAT,
            "velocity_config": asdict(self.velocity_cfg),
            "data_identity": self.data_identity,
            "robot_identity": self.robot_identity,
            "initialization": self.initialization,
            "collected": self.collected,
            "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        }
        temp = self.log_dir / (name + ".tmp")
        torch.save(ck, temp)
        temp.replace(self.log_dir / name)

    def load(self, path, checkpoint=None):
        ck = (
            checkpoint
            if checkpoint is not None
            else torch.load(path, map_location="cpu", weights_only=False)
        )
        if not self._velocity_ready:
            self._pending_resume = ck
            return
        super().load(path, checkpoint=ck)
        self.initialization, self.collected = ck["initialization"], ck["collected"]
        torch.set_rng_state(ck["rng"])
        torch.cuda.set_rng_state_all(ck["cuda_rng"])
        np.random.set_state(ck["numpy_rng"])
        random.setstate(ck["python_rng"])

    def run(self):
        try:
            super().run()
        except KeyboardInterrupt:
            self.save()
            print("[VELOCITY] Interrupted; saved ckpt_latest.pt", flush=True)
        finally:
            self.writer.close()
            self.env.close()
