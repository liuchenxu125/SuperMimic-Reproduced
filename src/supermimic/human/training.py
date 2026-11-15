"""Independent MorphMimic copy-student post-training, with the native actor frozen.

Unlike the original joint motion/human training loop, this task starts imagined
rollouts from paired reference states with a frozen motion-teacher warmup. It
does not collect simulator replay or update the world model/motion interface.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch

from supermimic.teleop.reference import active_sim_bundle, load_contract
from supermimic.training.numerics import require_finite
from supermimic.training.precision import configure_precision
from supermimic.training.schedule import cosine_lr
from supermimic.world_model.heads import AnalyticHeads, ReferenceFrame, flatten_history_term_major

from .constants import GOAL_K, HUMAN_GOAL_DIM, KP_GOAL_VERSION
from .data import PairedLibrary, read_pairs, split_pairs
from .features import sample_human_masks
from .loss_scales import TermScales
from .model import HumanEncoder, HumanPolicy
from .teacher import FrozenTeacher, digest, robot_state


@dataclass
class HumanTrainConfig:
    pairs: Path
    output_dir: Path
    teacher_ckpt: Path | None = None
    bundle: Path | None = None
    device: str = "cuda:0"
    iterations: int = 60_000
    batch_size: int = 256
    horizon: int = 32
    warmup_steps: int = 10
    lr: float = 2e-4
    lr_decay_iterations: int = 250_000
    lr_final_ratio: float = 0.03
    masking: Literal["mixed", "full", "five"] = "mixed"
    obs_noise: bool = True
    loss_normalization: bool = True
    action_weight: float = 1.0
    align_weight: float = 1.0
    validation_fraction: float = 0.1
    validate_every: int = 500
    log_every: int = 50
    save_every: int = 4000
    cache_clips: int = 4
    seed: int = 0
    cpu_threads: int = 1
    resume: Path | None = None


def data_identity(pairs, path):
    records = []
    for pair in pairs:
        records.append(
            {
                "name": pair.name,
                "robot_start_frame": pair.robot_start_frame,
                "human_start_frame": pair.human_start_frame,
                "length": pair.length,
                "files": [
                    {"path": str(p), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                    for p in (pair.robot_npz, pair.human_npz)
                ],
            }
        )
    import hashlib

    return {
        "manifest_sha256": digest(path),
        "files": records,
        "fingerprint": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
    }


def rollout_copy(teacher, human, robot, human_poses, cfg, *, update_norms):
    """Copy actions and latents at the human student's own detached imagined states."""
    batch = len(robot)
    device = robot.device
    xi = teacher.wm.spec.xi_offset.expand(batch, -1)
    state = robot_state(robot[:, 0])
    action_prev = torch.zeros(batch, 27, device=device)
    obs = teacher.observation(state, action_prev, xi, robot[:, 0, 3:7], cfg.obs_noise)
    hist = [obs] * teacher.pcfg.history
    # A warmup generates consistent history/previous actions without inventing
    # logged actions from reference joint positions. No original replay is changed.
    with torch.no_grad():
        for t in range(cfg.warmup_steps):
            packed = flatten_history_term_major(torch.stack(hist), num_joints=27)
            goal = teacher.motion_goal(robot[:, t : t + GOAL_K])
            action_prev = teacher.policy(packed, goal)
            state = teacher.wm.step(state, action_prev, xi)
            obs = teacher.observation(state, action_prev, xi, robot[:, t + 1, 3:7], cfg.obs_noise)
            hist = hist[1:] + [obs]
    points, frames, heading = sample_human_masks(
        batch, cfg.masking, device, heading_frac=1.0, preview_mix=False
    )
    action_error, latent_error, survival = [], [], torch.ones(batch, device=device)
    for step in range(cfg.horizon):
        t = cfg.warmup_steps + step
        packed = flatten_history_term_major(torch.stack(hist), num_joints=27).detach()
        with torch.no_grad():
            goal = teacher.motion_goal(robot[:, t : t + GOAL_K])
            z_teacher = teacher.policy.encoders["motion"](goal)
            target = teacher.policy.decoder(z_teacher, packed)
            anchor = teacher.anchor_quat(state)
        hw = human_poses[:, t : t + GOAL_K]
        human_goal = human.pack(
            hw[..., :3], hw[..., 3:], anchor, points, frames, heading, update=update_norms
        )
        z_human = human(human_goal)
        action = teacher.policy.decoder(z_human, packed)
        action_error.append((action - target).square().mean(-1))
        latent_error.append((z_human - z_teacher).square().mean(-1))
        # Copy mode deliberately has no BPTT graph and no critic. Only the current
        # human encoder is differentiated, while the student drives the WM.
        with torch.no_grad():
            state = teacher.wm.step(state.detach(), action.detach(), xi)
            ref = ReferenceFrame(None, None, None, None, robot[:, t, :3], robot[:, t, 3:7])
            feats = AnalyticHeads.termination_features(state[:, :3], state[:, 3:7], ref)
            survival *= 1 - teacher.wm.term_head(feats).sigmoid()
            obs = teacher.observation(
                state, action.detach(), xi, robot[:, t + 1, 3:7], cfg.obs_noise
            )
            hist = hist[1:] + [obs]
    terms = {"copy": torch.stack(action_error).mean(), "align": torch.stack(latent_error).mean()}
    return terms, {
        "imagined_survival": float(survival.mean()),
        "obs_dim": packed.shape[-1],
        "human_goal_dim": human_goal.shape[-1],
    }


class HumanTrainer:
    def __init__(self, cfg: HumanTrainConfig, root=None):
        self.cfg = cfg
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError(
                "Human post-training is single-process; do not launch it with torchrun"
            )
        if (
            min(
                cfg.iterations,
                cfg.batch_size,
                cfg.horizon,
                cfg.log_every,
                cfg.save_every,
                cfg.validate_every,
                cfg.cpu_threads,
                cfg.lr_decay_iterations,
            )
            < 1
            or cfg.warmup_steps < 10
        ):
            raise ValueError(
                "Training counts must be positive; warmup_steps must cover the 10-frame history"
            )
        if (
            cfg.action_weight < 0
            or cfg.align_weight < 0
            or cfg.action_weight + cfg.align_weight <= 0
        ):
            raise ValueError("Require nonnegative human loss weights, with at least one nonzero")
        if (
            not all(math.isfinite(x) for x in (cfg.lr, cfg.action_weight, cfg.align_weight))
            or cfg.lr <= 0
        ):
            raise ValueError("Learning rate and loss weights must be finite; lr must be positive")
        torch.set_num_threads(cfg.cpu_threads)
        device = torch.device(cfg.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        torch.manual_seed(cfg.seed)
        self.generator = torch.Generator().manual_seed(cfg.seed)
        self.val_generator = torch.Generator().manual_seed(cfg.seed + 1)
        root = Path(root or Path(__file__).resolve().parents[3])
        bundle = (cfg.bundle or active_sim_bundle(root)).resolve()
        checkpoint = cfg.teacher_ckpt or Path(load_contract(bundle)["checkpoint_path"])
        self.teacher = FrozenTeacher(checkpoint, bundle, cfg.device)
        self.precision = configure_precision(self.teacher.config.get("tf32", True))
        self.human = HumanEncoder(self.teacher.pcfg).to(device)
        self.policy = HumanPolicy(self.teacher.policy, self.human)
        self.opt = torch.optim.Adam(self.human.parameters(), lr=cfg.lr, betas=(0.9, 0.95))
        self.scales = TermScales(("copy", "align")).to(device)
        pairs = read_pairs(cfg.pairs)
        self.identity = data_identity(pairs, cfg.pairs)
        train, val = split_pairs(pairs, cfg.validation_fraction)
        window = cfg.warmup_steps + cfg.horizon + GOAL_K
        self.data = PairedLibrary(train, self.teacher.meta, window, cfg.cache_clips)
        self.validation = (
            PairedLibrary(val, self.teacher.meta, window, cfg.cache_clips) if val else None
        )
        self.split = {"train": [p.name for p in train], "validation": [p.name for p in val]}
        self.iteration = 0
        self.out = cfg.output_dir.resolve()
        if self.out == self.teacher.checkpoint.parent or self.out == self.teacher.bundle:
            raise ValueError(
                "Use a NEW output directory, separate from teacher training/deployment"
            )
        if self.out.exists() and any(self.out.iterdir()) and cfg.resume is None:
            raise FileExistsError("Human output already exists; use --resume or a new --output-dir")
        self._frozen_versions = [
            p._version for m in (self.teacher.policy, self.teacher.wm) for p in m.parameters()
        ]
        if cfg.resume:
            self.load(cfg.resume)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "human_config.json").write_text(json.dumps(asdict(cfg), default=str, indent=2))
        (self.out / "data_identity.json").write_text(json.dumps(self.identity, indent=2))
        (self.out / "split.json").write_text(json.dumps(self.split, indent=2))

    def assert_frozen(self):
        params = [p for m in (self.teacher.policy, self.teacher.wm) for p in m.parameters()]
        if [p._version for p in params] != self._frozen_versions or any(
            p.grad is not None for p in params
        ):
            raise RuntimeError("Frozen motion teacher/world model was modified")

    def step(self):
        self.human.train()
        robot, poses = self.data.sample(self.cfg.batch_size, self.generator, self.cfg.device)
        self.opt.zero_grad(set_to_none=True)
        terms, info = rollout_copy(
            self.teacher, self.human, robot, poses, self.cfg, update_norms=True
        )
        scale = self.scales.scales() if self.cfg.loss_normalization else {k: 1 for k in terms}
        loss = (
            self.cfg.action_weight * terms["copy"] / scale["copy"]
            + self.cfg.align_weight * terms["align"] / scale["align"]
        )
        require_finite(loss, "human copy loss")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(self.human.parameters(), 1.0, error_if_nonfinite=True)
        torch.nn.utils.clip_grad_value_(self.human.parameters(), 0.1)
        self.opt.step()
        self.scales.update(terms)
        self.assert_frozen()
        return {
            "loss": float(loss.detach()),
            "action_mse": float(terms["copy"].detach()),
            "latent_mse": float(terms["align"].detach()),
            "grad_norm": float(grad),
            **info,
        }

    @torch.no_grad()
    def validate(self):
        if self.validation is None:
            return {"validation": "no held-out clip families"}
        self.human.eval()
        robot, poses = self.validation.sample(
            self.cfg.batch_size, self.val_generator, self.cfg.device
        )
        terms, info = rollout_copy(
            self.teacher, self.human, robot, poses, self.cfg, update_norms=False
        )
        return {
            "val_action_mse": float(terms["copy"]),
            "val_latent_mse": float(terms["align"]),
            **info,
        }

    def save(self, name):
        self.assert_frozen()
        checkpoint = {
            "format": "supermimic-human-copy-v1",
            "human_goal_version": KP_GOAL_VERSION,
            "human_goal_dim": HUMAN_GOAL_DIM,
            "obs_dim": 930,
            "motion_goal_dim": 690,
            "iteration": self.iteration,
            "config": asdict(self.cfg),
            "teacher_checkpoint": str(self.teacher.checkpoint),
            "teacher_sha256": self.teacher.sha256,
            "bundle": str(self.teacher.bundle),
            "contract_sha256": self.teacher.contract_sha256,
            "model_sha256": self.teacher.model_sha256,
            "human": self.human.state_dict(),
            "optimizer": self.opt.state_dict(),
            "loss_scales": self.scales.state_dict(),
            "data_identity": self.identity,
            "split": self.split,
            "precision": self.precision,
            "rng": torch.get_rng_state(),
            "sampler_rng": self.generator.get_state(),
            "val_sampler_rng": self.val_generator.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        target = self.out / name
        temporary = target.with_suffix(target.suffix + ".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(target)

    def load(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if (
            ck.get("format") != "supermimic-human-copy-v1"
            or ck.get("human_goal_version") != KP_GOAL_VERSION
        ):
            raise ValueError(
                "Not a compatible human checkpoint; never resume sm-train with this file"
            )
        if (
            ck["teacher_sha256"] != self.teacher.sha256
            or ck["data_identity"] != self.identity
            or ck["split"] != self.split
        ):
            raise ValueError("Human resume teacher/data/split mismatch")
        if (
            ck["contract_sha256"] != self.teacher.contract_sha256
            or ck["model_sha256"] != self.teacher.model_sha256
        ):
            raise ValueError("Human resume robot contract/model changed")
        mutable = {
            "iterations",
            "output_dir",
            "resume",
            "teacher_ckpt",
            "bundle",
            "device",
            "log_every",
            "save_every",
            "validate_every",
        }
        for key, value in asdict(self.cfg).items():
            if key not in mutable and str(value) != str(ck["config"][key]):
                raise ValueError(f"Human resume configuration changed: {key}")
        self.human.load_state_dict(ck["human"])
        self.opt.load_state_dict(ck["optimizer"])
        self.scales.load_state_dict(ck["loss_scales"])
        self.iteration = int(ck["iteration"])
        torch.set_rng_state(ck["rng"])
        self.generator.set_state(ck["sampler_rng"])
        self.val_generator.set_state(ck["val_sampler_rng"])
        if ck["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ck["cuda_rng"])

    def run(self):
        print(
            json.dumps(
                {
                    "task": "human-copy",
                    "teacher": str(self.teacher.checkpoint),
                    "teacher_sha256": self.teacher.sha256,
                    "obs_dim": 930,
                    "motion_goal_dim": 690,
                    "human_goal_dim": HUMAN_GOAL_DIM,
                    "train_pairs": len(self.data.pairs),
                    "validation_pairs": len(self.split["validation"]),
                    "original_training_modified": False,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        started = time.monotonic()
        try:
            while self.iteration < self.cfg.iterations:
                lr = cosine_lr(
                    self.iteration,
                    self.cfg.lr,
                    self.cfg.lr_final_ratio,
                    self.cfg.lr_decay_iterations,
                )
                for group in self.opt.param_groups:
                    group["lr"] = lr
                metrics = self.step()
                self.iteration += 1
                if self.iteration % self.cfg.validate_every == 0:
                    metrics.update(self.validate())
                if (
                    self.iteration == 1
                    or self.iteration % self.cfg.log_every == 0
                    or self.iteration % self.cfg.validate_every == 0
                ):
                    record = {
                        "iteration": self.iteration,
                        "seconds": time.monotonic() - started,
                        "lr": lr,
                        **metrics,
                    }
                    print(json.dumps(record), flush=True)
                    with (self.out / "metrics.jsonl").open("a") as f:
                        f.write(json.dumps(record) + "\n")
                if self.iteration % self.cfg.save_every == 0:
                    self.save(f"human_{self.iteration:07d}.pt")
                    self.save("human_latest.pt")
        except KeyboardInterrupt:
            print("Human training interrupted; saving human_latest.pt", flush=True)
            self.save("human_latest.pt")
        else:
            self.save("human_latest.pt")
