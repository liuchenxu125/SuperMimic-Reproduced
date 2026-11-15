"""Independent motion + human joint task; the original Trainer is not edited.

Reuses native simulator, recorder, WM loss, motion BPTT/critic, and update-based
schedule. Adds mixed-interface collection and a replay-based human copy update.
Single Casbot02 / single GPU, matching MorphMimic's recommended copy recipe.
"""
from __future__ import annotations

import copy
import json
import os
import random
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path

import numpy as np
import torch

from supermimic.config import DataCfg, TrainConfig
from supermimic.robots import get_robot
from supermimic.robots.profiles import CASBOT_ASSETS
from supermimic.training.loop import Trainer
from .constants import HUMAN_GOAL_DIM, KP_GOAL_VERSION
from .data import PairedLibrary, read_pairs, split_pairs
from .features import sample_human_masks
from .joint_data import HumanWindows, stage_robot_pairs
from .joint_student import JointCopyStep, JointPolicyUpdates, TorsoFrame, frozen
from .model import HumanEncoder
from .teacher import FrozenTeacher, digest
from .training import HumanTrainConfig, data_identity, rollout_copy

FORMAT = "supermimic-motion-human-copy-v1"


def require_prepared_pairs(path):
    report_path = Path(path).resolve().parent/"preparation_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report.get("requested_limit") == 0 and report.get("pending", 0) > 0:
            raise ValueError("Full SEED pairing is still incomplete; wait for preparation pending=0 before training")


def check_joint_resume_config(saved, requested):
    """Batch size may change explicitly without resetting weights/Adam state.

    It changes sampling/gradient variance, so record it. Keep masks, temporal
    semantics, data selection and other algorithm settings strictly protected.
    """
    changed = {key: [saved.get(key), requested.get(key)]
               for key in saved.keys() | requested.keys()
               if saved.get(key) != requested.get(key)}
    if set(changed) - {"human_batch"}:
        raise ValueError("Joint resume config changed: " + ", ".join(sorted(set(changed)-{"human_batch"})))
    return changed


def restore_config(values):
    """Reconstruct the native nested dataclasses without changing global defaults."""
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
    return fill(TrainConfig(), copy.deepcopy(values))


def robot_contract():
    profile = get_robot("casbot02")
    model = profile.model(collision_model="primitive")
    params = profile.motor_parameters()
    names = [model.joint(i).name for i in range(1, model.njnt)]
    meta = {"joint_names": names, "root_body_name": "base_link", "motors": []}
    for name in names:
        p = params["motors"][name]
        meta["motors"].append({"name": name, "q0": params["default_joint_pos"][name],
            "kp": p["stiffness"], "kd": p["damping"], "scale": .25*p["effort_limit"]/p["stiffness"],
            "effort_limit_nm": p["effort_limit"], "armature": p["armature"]})
    meta["assets"] = {name: digest(CASBOT_ASSETS/name) for name in
                       ("motor_parameters.json", "xml/casbot02.xml", "xml/casbot02_primitives.xml")}
    return meta, model


def check_warmstart_contract(meta, exported):
    if tuple(meta["joint_names"]) != tuple(exported["joint_names"]):
        raise ValueError("Warm-start joint order differs from current Casbot02")
    for a, b in zip(meta["motors"], exported["motors"]):
        for key in ("q0", "kp", "kd", "scale", "effort_limit_nm", "armature"):
            if not np.isclose(a[key], b[key], rtol=1e-6, atol=1e-6):
                raise ValueError(f"Warm-start motor contract differs: {a['name']} {key}; use matching assets or --from-scratch")


class LiveTeacher:
    """Reuse existing teacher feature math, bound to the *updating* joint networks."""
    motion_goal = FrozenTeacher.motion_goal
    observation = FrozenTeacher.observation

    def __init__(self, trainer, torso):
        self.policy, self.wm, self.pcfg = trainer.policy, trainer.wm, trainer.cfg.policy
        self.q0 = trainer.lib.default_joint_pos
        self.noise_half = trainer.human_step.noise_half
        self.anchor_quat = torso


class JointTrainer(Trainer):
    def __init__(self, cfg, *, pairs, human_batch=512, human_fraction=.5,
                 masking="mixed", cache_clips=0, validation_fraction=0.0,
                 validate_every=500, init_checkpoint=None, init_bundle=None,
                 init_human=None):
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("Independent joint task currently supports one GPU/process")
        if cfg.robot != "casbot02" or cfg.env.anchor_obs or not cfg.env.anchor_ori_obs:
            raise ValueError("Joint task preserves Casbot02 no-position + per-history rot6d observations")
        if cfg.policy.history != 10 or cfg.env.future_frames != 10:
            raise ValueError("Joint task requires the existing 10-frame history/preview")
        if cfg.world_model.state_spec.dt != .02:
            raise ValueError("Joint task requires the native 50 Hz dynamics/control clock")
        if not 0 < human_fraction < 1 or cfg.env.num_envs < 2:
            raise ValueError("Need both interfaces: num_envs >= 2 and 0 < human_fraction < 1")
        if human_batch < 1 or validate_every < 1 or cache_clips < 0:
            raise ValueError("Invalid human batch/validation interval/cache")
        self._joint_ready = False
        self._pending_checkpoint = None
        self.meta, model = robot_contract()
        pairs_path = Path(pairs).resolve()
        require_prepared_pairs(pairs_path)
        all_pairs = read_pairs(pairs_path)
        identity = data_identity(all_pairs, pairs_path)
        train_pairs, val_pairs = split_pairs(all_pairs, validation_fraction)
        self.pair_identity = identity
        self.pair_split = {"train": [p.name for p in train_pairs], "validation": [p.name for p in val_pairs]}
        self.joint_config = dict(pairs=str(pairs_path), human_batch=human_batch,
            human_fraction=human_fraction, masking=masking, cache_clips=cache_clips,
            validation_fraction=validation_fraction, validate_every=validate_every,
            student="copy", preview_mix=False, jitter=False, drift=False,
            human_goal_version=KP_GOAL_VERSION, horizon=cfg.policy_trainer.horizon,
            history=cfg.policy.history, future_frames=cfg.env.future_frames,
            obs_noise=cfg.policy_trainer.obs_noise)
        self.validate_every = validate_every
        run_dir = Path(cfg.log_root)/cfg.run_name
        resume = None if cfg.resume is None else torch.load(cfg.resume, map_location="cpu", weights_only=False)
        self.resume_resource_overrides = {}
        if resume is not None:
            if resume.get("joint_format") != FORMAT:
                raise ValueError("--resume requires a joint checkpoint; use --init-checkpoint for motion-only weights")
            self.resume_resource_overrides = check_joint_resume_config(resume["joint_config"], self.joint_config)
            if resume["pair_identity"] != identity or resume["pair_split"] != self.pair_split:
                raise ValueError("Joint resume paired data/config/split changed")
            if resume["robot_contract"] != self.meta:
                raise ValueError("Joint resume robot assets/motor contract changed")
            old_batch = resume["config"]["policy_trainer"]["batch_size"]
            if old_batch != cfg.policy_trainer.batch_size:
                self.resume_resource_overrides["motion_batch"] = [old_batch, cfg.policy_trainer.batch_size]
            old_wm_batch = resume["config"]["wm_trainer"]["batch_size"]
            if old_wm_batch != cfg.wm_trainer.batch_size:
                self.resume_resource_overrides["wm_batch"] = [old_wm_batch, cfg.wm_trainer.batch_size]
            if self.resume_resource_overrides:
                print("[JOINT] resume batch changes: " + json.dumps(self.resume_resource_overrides), flush=True)
        elif (run_dir/"config.json").exists():
            raise FileExistsError("Joint run exists; use --resume or a new --run-name")
        self.initialization = {"mode": "scratch"}
        initial = None
        if init_checkpoint is not None and resume is None:
            from supermimic.teleop.reference import load_contract
            exported = load_contract(init_bundle)
            source_sha = digest(init_checkpoint)
            if source_sha != exported["checkpoint_sha256"]:
                raise ValueError("Initialization checkpoint and bundle do not match")
            check_warmstart_contract(self.meta, exported)
            initial = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
            saved = initial["config"]
            pc = saved["policy"]
            if (saved.get("robot") != "casbot02" or saved["env"].get("anchor_obs") or
                not saved["env"].get("anchor_ori_obs") or
                (pc["num_joints"], pc["history"], pc["goal_dim"], len(pc["obs_term_scale"])) != (27,10,690,6)):
                raise ValueError("Motion initialization must already have the Casbot02 930/690/27 orientation contract")
            self.initialization = {"mode": "motion-weights-only", "source": str(Path(init_checkpoint).resolve()),
                                   "source_sha256": source_sha, "source_iteration": int(initial["iteration"])}
        stage_robot_pairs(train_pairs, run_dir/"paired_robot", self.meta)
        cfg.data = DataCfg(npz_dir=run_dir/"paired_robot")
        super().__init__(cfg)
        if self.world != 1:
            raise ValueError("Joint task is single-GPU")
        if initial is not None:
            # A new dataset/interface is a new run: no old sampler or optimizers,
            # no inherited 96000 iteration counter. Native resume is separate.
            self.wm.load_state_dict(initial["wm"])
            self.policy.load_state_dict(initial["policy"])
            self.critic.load_state_dict(initial["critic"])
        self.human = HumanEncoder(cfg.policy).to(self.device)
        self.windows = HumanWindows(train_pairs, self.meta, self.lib, cache_clips)
        self.torso = TorsoFrame(model, self.device)
        self.human_step = JointCopyStep(self.wm, self.policy, self.human, self.lib,
            self.windows, self.torso, batch_size=human_batch, horizon=cfg.policy_trainer.horizon,
            masking=masking, lr=cfg.policy_trainer.lr, obs_noise=cfg.policy_trainer.obs_noise)
        self.policy_trainer = JointPolicyUpdates(self.policy_trainer, self.human_step)
        if init_human is not None and resume is None:
            h = torch.load(init_human, map_location="cpu", weights_only=False)
            if (h.get("format") != "supermimic-human-copy-v1" or
                h.get("teacher_sha256") != self.initialization.get("source_sha256") or
                h.get("human_goal_version") != KP_GOAL_VERSION):
                raise ValueError("Human initialization must match the motion checkpoint and v10 features")
            self.human.load_state_dict(h["human"])
            self.initialization["human_source"] = str(Path(init_human).resolve())
        self.human_env_count = max(1, min(cfg.env.num_envs-1, round(cfg.env.num_envs*human_fraction)))
        self._human_env = torch.arange(cfg.env.num_envs, device=self.device) < self.human_env_count
        self._points = torch.empty((cfg.env.num_envs,24), dtype=torch.bool, device=self.device)
        self._times = torch.empty((cfg.env.num_envs,10), dtype=torch.bool, device=self.device)
        self._heading = torch.empty(cfg.env.num_envs, dtype=torch.bool, device=self.device)
        self.draw_masks(torch.arange(cfg.env.num_envs, device=self.device))
        if val_pairs:
            print(f"[JOINT] loading {len(val_pairs)} held-out pairs",flush=True)
        self.validation = PairedLibrary(val_pairs, self.meta, 52, cache_clips) if val_pairs else None
        self.val_generator = torch.Generator().manual_seed(cfg.seed+1)
        self.live_teacher = LiveTeacher(self, self.torso)
        self.collection_counts = {"motion": 0, "human": 0}
        self._initial_versions = {name: [p._version for p in module.parameters()] for name, module in
                                   (("wm", self.wm), ("motion", self.policy), ("human", self.human))}
        self._joint_ready = True
        if self._pending_checkpoint is not None:
            self.restore_joint(self._pending_checkpoint)
            self._pending_checkpoint = None
        self._reported_update = self.policy_trainer.human_updates
        self._initial_versions = {name: [p._version for p in module.parameters()] for name, module in
                                   (("wm", self.wm), ("motion", self.policy), ("human", self.human))}
        (self.log_dir/"joint_config.json").write_text(json.dumps(self.joint_config, indent=2))
        (self.log_dir/"joint_runtime.json").write_text(json.dumps({
            "compile_mode": cfg.policy_trainer.compile,
            "inductor_compile_threads": os.environ.get("TORCHINDUCTOR_COMPILE_THREADS"),
            "human_storage": "contiguous_cpu" if cache_clips == 0 else "bounded_cpu_cache",
            "cpu_threads": torch.get_num_threads(),
            "resume_batch_changes": self.resume_resource_overrides}, indent=2))
        (self.log_dir/"robot_contract.json").write_text(json.dumps(self.meta, indent=2))
        (self.log_dir/"split.json").write_text(json.dumps(self.pair_split, ensure_ascii=False, indent=2))
        print("[JOINT] " + json.dumps({"motion_envs": cfg.env.num_envs-self.human_env_count,
            "human_envs": self.human_env_count, "obs": 930, "motion_goal": 690,
            "human_goal": HUMAN_GOAL_DIM, "actions": 27, "student": "copy",
            "train_pairs": len(train_pairs), "validation_pairs": len(val_pairs),
            "initialization": self.initialization}, ensure_ascii=False), flush=True)

    @torch.no_grad()
    def draw_masks(self, ids):
        if not len(ids):
            return
        p, t, h = sample_human_masks(len(ids), self.joint_config["masking"], self.device,
                                    heading_frac=1., preview_mix=False)
        self._points[ids], self._times[ids], self._heading[ids] = p, t, h

    @torch.no_grad()
    def act(self, obs):
        # Keep the native observation layout/scales and native motion path exact.
        motion = self.policy(obs["policy"], obs["goal"])
        poses = self.windows.future(self.command.frame, 10)
        state = self.recorder.io.read()
        self.human.train()
        goal = self.human.pack(poses[..., :3], poses[..., 3:], self.torso(state),
                               self._points, self._times, self._heading, update=True)
        action = self.policy.decoder(self.human(goal), obs["policy"])
        return torch.where(self._human_env[:, None], action, motion)

    def _collect_step(self):
        super()._collect_step()
        self.collection_counts["human"] += self.human_env_count
        self.collection_counts["motion"] += self.cfg.env.num_envs-self.human_env_count
        self.draw_masks(self.env.reset_buf.nonzero().flatten())

    @torch.no_grad()
    def validate_human(self):
        if self.validation is None:
            return {}
        self.human.eval()
        robot, poses = self.validation.sample(self.joint_config["human_batch"], self.val_generator, self.device)
        cfg = HumanTrainConfig(pairs=Path(self.joint_config["pairs"]), output_dir=self.log_dir,
                horizon=self.cfg.policy_trainer.horizon, masking=self.joint_config["masking"],
                obs_noise=self.cfg.policy_trainer.obs_noise)
        with frozen(self.wm, self.policy):
            terms, info = rollout_copy(self.live_teacher, self.human, robot, poses, cfg, update_norms=False)
        self.human.train()
        return {"val_action_mse": float(terms["copy"]), "val_latent_mse": float(terms["align"]),
                "val_imagined_survival": info["imagined_survival"]}

    def _synchronize(self):
        # Parent calls this after updates. Report/validate once per completed
        # update count, while keeping the original main-loop schedule unchanged.
        super()._synchronize()
        if not getattr(self, "_joint_ready", False):
            return
        count = self.policy_trainer.human_updates
        if count and count != getattr(self, "_reported_update", -1) and (
                count % self.cfg.schedule.log_interval == 0 or count == 1):
            self._reported_update = count
            record = {"joint_update": count, "outer_update": self.iteration+1,
                      **self.policy_trainer.latest, "collected": self.collection_counts.copy()}
            if count % self.validate_every == 0:
                record.update(self.validate_human())
            record["modules_updated"] = {name: self._initial_versions[name] != [p._version for p in module.parameters()]
                for name, module in (("wm", self.wm), ("motion", self.policy), ("human", self.human))}
            with (self.log_dir/"joint_metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record)+"\n")
            print("[JOINT] " + json.dumps(record), flush=True)
            for key, value in record.items():
                if isinstance(value, (int,float)):
                    self.writer.add_scalar(f"human/{key}", value, self.iteration+1)

    def save(self, name="ckpt_latest.pt"):
        # Native motion checkpoint keys remain compatible with ordinary motion
        # evaluation/export; the extra joint entry is required to resume human.
        ck = {"checkpoint_version": 2, "iteration_unit": "update", "iteration": self.iteration,
            "num_updates": self.num_updates, "env_steps": self.env_steps,
            "replay_refill_env_steps": self.replay_refill_env_steps,
            "collection_accumulator": self._collection_acc, "train_seconds": self._train_seconds,
            "precision": self.precision, "resume_migration": self.resume_migration,
            "wm": self.wm.state_dict(), "policy": self.policy.state_dict(), "critic": self.critic.state_dict(),
            "wm_opt": self.wm_trainer.opt.state_dict(), "policy_opt": self.policy_trainer.opt.state_dict(),
            "critic_opt": self.policy_trainer.critic_opt.state_dict(),
            "bin_return": self.command.bin_return, "bin_memory": self.command.bin_memory,
            "config": asdict(self.cfg), "joint_format": FORMAT, "joint_config": self.joint_config,
            "pair_identity": self.pair_identity, "pair_split": self.pair_split, "robot_contract": self.meta,
            "initialization": self.initialization, "human": self.human.state_dict(),
            "resume_batch_changes": self.resume_resource_overrides,
            "human_opt": self.human_step.opt.state_dict(), "human_scales": self.human_step.scales.state_dict(),
            "joint_counts": [self.policy_trainer.motion_updates, self.policy_trainer.human_updates],
            "collection_counts": self.collection_counts,
            "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
            "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
            "val_rng": self.val_generator.get_state()}
        temporary = self.log_dir/(name+".tmp")
        torch.save(ck, temporary)
        temporary.replace(self.log_dir/name)

    def load(self, path, checkpoint=None):
        ck = checkpoint if checkpoint is not None else torch.load(path, map_location="cpu", weights_only=False)
        if not self._joint_ready:
            self._pending_checkpoint = ck
        else:
            self.restore_joint(ck)

    def restore_joint(self, ck):
        Trainer.load(self, self.cfg.resume, checkpoint=ck)
        self.human.load_state_dict(ck["human"])
        self.human_step.opt.load_state_dict(ck["human_opt"])
        self.human_step.scales.load_state_dict(ck["human_scales"])
        self.policy_trainer.motion_updates, self.policy_trainer.human_updates = ck["joint_counts"]
        self.collection_counts = ck["collection_counts"]
        self.initialization = ck["initialization"]
        torch.set_rng_state(ck["rng"])
        torch.cuda.set_rng_state_all(ck["cuda_rng"])
        np.random.set_state(ck["numpy_rng"])
        random.setstate(ck["python_rng"])
        self.val_generator.set_state(ck["val_rng"])

    def run(self):
        try:
            super().run()
        except KeyboardInterrupt:
            print("[JOINT] Interrupted: saving joint ckpt_latest.pt", flush=True)
            self.save()
        finally:
            self.writer.close()
            self.env.close()
