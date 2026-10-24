"""SuperMimic training loop (paper Algorithm A.1): collect -> fit world model -> improve policy."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from supermimic.config import TrainConfig
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.imagination.policy_trainer import PolicyTrainer
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.policy.nets import Critic, Policy
from supermimic.robots import get_robot
from supermimic.training.checkpoints import checkpoint_progress
from supermimic.training.distributed import (
    allreduce_mean,
    broadcast_params,
    finalize,
    init_distributed,
)
from supermimic.training.precision import configure_precision
from supermimic.training.schedule import collection_steps, cosine_lr
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.wm_trainer import WorldModelTrainer
from supermimic.world_model.world_model import WorldModel


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        profile = get_robot(cfg.robot)
        cfg.policy.num_joints = profile.num_joints
        cfg.world_model.state_spec.num_joints = profile.num_joints
        if cfg.world_model.state_spec.xi_offset == (0.75,) + (0.0,) * 32:
            cfg.world_model.state_spec.xi_offset = (0.75,) + (0.0,) * (profile.num_joints + 3)
        if cfg.world_model.state_spec.xi_scale == (0.45,) + (0.01,) * 29 + (0.05,) * 3:
            cfg.world_model.state_spec.xi_scale = (
                (0.45,) + (0.01,) * profile.num_joints + (0.05,) * 3
            )
        cfg.policy.goal_dim = cfg.env.future_frames * profile.layout.goal_dim + (
            9 if cfg.env.anchor_obs else 0
        )
        cfg.policy_trainer.anchor_obs = cfg.env.anchor_obs
        cfg.policy_trainer.future_frames = cfg.env.future_frames
        cfg.policy_trainer.history = cfg.policy.history
        sch = cfg.schedule
        if (
            min(
                sch.total_iterations,
                sch.log_interval,
                sch.checkpoint_interval,
                sch.lr_decay_iterations,
                sch.policy_updates_per_wm_update,
            )
            <= 0
        ):
            raise ValueError("Training lengths and intervals must be positive")
        collection_steps(0.0, sch.updates_per_env_step)
        if sch.warmup_iterations < max(cfg.wm_trainer.window + 1, cfg.policy.history):
            raise ValueError("warmup_iterations must fill replay windows and observation history")
        self.log_dir = Path(cfg.log_root) / cfg.run_name
        resume_checkpoint = None
        if cfg.resume is not None:
            resume_checkpoint = torch.load(cfg.resume, map_location="cpu", weights_only=False)
            progress = checkpoint_progress(resume_checkpoint, allow_legacy=cfg.legacy_resume)
            if progress.migrated_legacy:
                if self.log_dir.resolve() == cfg.resume.resolve().parent:
                    raise ValueError(
                        "Legacy migration requires a NEW --run-name; keep old logs intact"
                    )
            else:
                saved = resume_checkpoint["config"]
                for key in (
                    "updates_per_env_step",
                    "policy_updates_per_wm_update",
                    "lr_decay_iterations",
                    "lr_final_ratio",
                ):
                    if saved["schedule"][key] != getattr(sch, key):
                        raise ValueError(f"Resume schedule mismatch: {key}; use the saved setting")
                if saved.get("tf32", True) != cfg.tf32:
                    raise ValueError("Resume TF32 setting differs from checkpoint")
            if resume_checkpoint["config"].get("robot", "g1") != cfg.robot:
                raise ValueError("Resume robot differs from checkpoint")
            if resume_checkpoint["config"]["policy"]["goal_dim"] != cfg.policy.goal_dim:
                raise ValueError("Resume goal/anchor dimensions differ from checkpoint")
        if (self.log_dir / "config.json").exists() and (
            cfg.resume is None or self.log_dir.resolve() != cfg.resume.resolve().parent
        ):
            raise FileExistsError(
                "Run already exists; resume its own checkpoint or use a new --run-name"
            )
        self.precision = configure_precision(cfg.tf32)
        self.rank, local_rank, self.world = init_distributed()
        self.is_main = self.rank == 0
        if self.world > 1:
            cfg.device = f"cuda:{local_rank}"
            # Paper: lr x sqrt(2) for 2 GPUs (each GPU keeps its own 4096 envs and buffer).
            cfg.wm_trainer.lr *= self.world**0.5
            cfg.policy_trainer.lr *= self.world**0.5
            cfg.policy_trainer.critic_lr *= self.world**0.5
        torch.manual_seed(cfg.seed + 1000 * self.rank)
        self.device = cfg.device
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir / "tb")) if self.is_main else None

        if cfg.data.library is not None:
            files = [cfg.data.library]
            cfg.env.compact_library = True
        else:
            files = discover_npz(
                cfg.data.npz_dir,
                limit=cfg.data.limit,
                names=cfg.data.clips,
                manifest=cfg.data.manifest,
            )
        if not files:
            raise FileNotFoundError(f"no clips under {cfg.data.npz_dir}")
        self.env = make_env(
            SuperMimicEnvCfgArgs(
                npz_files=[str(f) for f in files],
                robot=cfg.robot,
                num_envs=cfg.env.num_envs,
                future_frames=cfg.env.future_frames,
                history_length=cfg.policy.history,
                episode_length_s=cfg.env.episode_length_s,
                action_noise_std=cfg.env.action_noise_std,
                obs_noise=cfg.env.obs_noise,
                reset_noise=cfg.env.reset_noise,
                domain_randomization=cfg.env.domain_randomization,
                sampling_mode=cfg.env.sampling_mode,
                replay_capacity=cfg.env.replay_capacity,
                compact_library=cfg.env.compact_library,
                anchor_obs=cfg.env.anchor_obs,
                seed=cfg.seed + 1000 * self.rank,
            ),
            device=self.device,
            tf32=cfg.tf32,
        )
        env = self.env
        self.command = env.command_manager.get_term("motion")
        self.lib = self.command.library
        self.recorder = env.recorder_manager.get_term("replay")
        self.buffer = self.recorder.buffer
        robot = env.scene["robot"]

        fk = G1Kinematics(profile.model(), device=self.device)
        heads = AnalyticHeads(
            fk,
            profile.tracked_bodies,
            profile.anchor_body,
            profile.root_body,
            robot.data.soft_joint_pos_limits[0].clone(),
            robot.data.default_joint_pos[0].clone(),
        )
        self.wm = WorldModel(cfg.world_model, heads, self.device)
        self.policy = Policy(cfg.policy).to(self.device)
        self.critic = Critic(cfg.critic, pcfg=cfg.policy).to(self.device)
        for m in (self.wm, self.policy, self.critic):
            broadcast_params(m)
        self.wm_trainer = WorldModelTrainer(self.wm, self.lib, cfg.wm_trainer)
        self.policy_trainer = PolicyTrainer(
            self.wm, self.policy, self.critic, self.lib, cfg.policy_trainer
        )
        n_params = lambda m: sum(p.numel() for p in m.parameters())
        if self.is_main:
            print(
                f"[INFO] params: world model {n_params(self.wm) / 1e6:.2f}M, policy {n_params(self.policy) / 1e6:.2f}M, critic {n_params(self.critic) / 1e6:.3f}M"
            )

        # Episode statistics of the training env (paper's "training return").
        self.ep_reward_sum = torch.zeros(env.num_envs, device=self.device)
        self.ep_len = torch.zeros(env.num_envs, device=self.device)
        self.max_step_reward = heads.reward_cfg.max_step_reward
        self.iteration = 0
        self.num_updates = 0
        self.env_steps = 0
        self.replay_refill_env_steps = 0
        self._collection_acc = 0.0
        self._time_offset = self._train_seconds = 0.0
        self.resume_migration = None
        if cfg.resume is not None:
            self.load(cfg.resume, checkpoint=resume_checkpoint)
        if self.is_main:
            (self.log_dir / "config.json").write_text(
                json.dumps(asdict(cfg), indent=2, default=str)
            )
            (self.log_dir / "runtime.json").write_text(
                json.dumps(
                    {
                        "iteration_unit": "update",
                        "precision": self.precision,
                        "resume_migration": self.resume_migration,
                    },
                    indent=2,
                )
            )
            print(
                f"[INFO] iteration_unit=update; precision={json.dumps(self.precision)}", flush=True
            )

    # ------------------------------------------------------------------ checkpoints

    def save(self, name: str = "ckpt_latest.pt") -> None:
        if not self.is_main:
            return
        target = self.log_dir / name
        temporary = target.with_suffix(target.suffix + ".tmp")
        torch.save(
            {
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
            },
            temporary,
        )
        temporary.replace(target)

    def load(self, path: Path, checkpoint: dict | None = None) -> None:
        ck = (
            checkpoint
            if checkpoint is not None
            else torch.load(path, map_location="cpu", weights_only=False)
        )
        progress = checkpoint_progress(ck, allow_legacy=self.cfg.legacy_resume)
        saved_robot = ck["config"].get("robot", "g1")
        if saved_robot != self.cfg.robot:
            raise ValueError(f"Checkpoint is for {saved_robot}, current robot is {self.cfg.robot}")
        self.wm.load_state_dict(ck["wm"])
        self.policy.load_state_dict(ck["policy"])
        self.critic.load_state_dict(ck["critic"])
        self.wm_trainer.opt.load_state_dict(ck["wm_opt"])
        self.policy_trainer.opt.load_state_dict(ck["policy_opt"])
        self.policy_trainer.critic_opt.load_state_dict(ck["critic_opt"])
        self.command.bin_return.copy_(ck["bin_return"])
        self.command.bin_memory.copy_(ck["bin_memory"])
        self.iteration = progress.iteration
        self.env_steps = progress.env_steps
        self.num_updates = progress.policy_updates
        self._collection_acc = progress.collection_accumulator
        self._time_offset = self._train_seconds = progress.train_seconds
        self.replay_refill_env_steps = int(ck.get("replay_refill_env_steps", 0))
        self.resume_migration = ck.get("resume_migration")
        if progress.migrated_legacy:
            self.resume_migration = {
                "source": str(path),
                "source_iteration_unit": "env_step",
                "source_iteration": ck["iteration"],
                "outer_updates": self.iteration,
                "old_lr": self.wm_trainer.opt.param_groups[0]["lr"],
                "next_lr": cosine_lr(
                    self.iteration,
                    self.cfg.wm_trainer.lr,
                    self.cfg.schedule.lr_final_ratio,
                    self.cfg.schedule.lr_decay_iterations,
                ),
                "note": "Explicit migration adopts update-based LR, sum actor loss and configured precision; not from-scratch reproduction",
            }
            print(f"[WARN] legacy migration: {json.dumps(self.resume_migration)}", flush=True)
        print(f"[INFO] resumed from {path}: update {self.iteration}, env_steps {self.env_steps}")

    # ------------------------------------------------------------------ main loop

    @torch.no_grad()
    def act(self, obs) -> torch.Tensor:
        return self.policy(obs["policy"], obs["goal"])

    def _synchronize(self) -> None:
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()

    def _collect_step(self) -> None:
        action = self.act(self._obs)
        self._obs, rew, term, tout, _extras = self.env.step(action)
        self.env_steps += 1
        self.ep_reward_sum += rew
        self.ep_len += 1
        done = term | tout
        if done.any():
            ids = done.nonzero().squeeze(-1)
            lens = self.ep_len[ids]
            norm_ret = self.ep_reward_sum[ids] / (self.max_step_reward * lens.clamp_min(1))
            self._finished_returns.extend(norm_ret.tolist())
            self._finished_lens.extend(lens.tolist())
            self.ep_reward_sum[ids] = 0.0
            self.ep_len[ids] = 0.0

    def run(self) -> None:
        cfg = self.cfg
        sch = cfg.schedule
        self._obs, _ = self.env.reset()
        t_start = time.time()
        timers = {"sim": 0.0, "wm": 0.0, "policy": 0.0}
        agg: dict[str, float] = {}
        agg_n = 0
        self._finished_returns: list[float] = []
        self._finished_lens: list[float] = []
        last_log = time.time()
        # Replay is not checkpointed. Refill after both a fresh start and a resume,
        # without counting these collection-only steps as training iterations.
        if self.iteration < sch.total_iterations:
            for _ in range(sch.warmup_iterations):
                self._collect_step()
                self.replay_refill_env_steps += 1

        while self.iteration < sch.total_iterations:
            # ---------------- (1) collect 1/UTD vectorised steps per outer update
            t0 = time.perf_counter()
            steps, self._collection_acc = collection_steps(
                self._collection_acc, sch.updates_per_env_step
            )
            for _ in range(steps):
                self._collect_step()
            self._synchronize()
            timers["sim"] += time.perf_counter() - t0

            # ---------------- (2)+(3) one WM update, then actor/critic updates
            lr = cosine_lr(
                self.iteration, cfg.wm_trainer.lr, sch.lr_final_ratio, sch.lr_decay_iterations
            )
            self.wm_trainer.set_lr(lr)
            self.policy_trainer.set_lr(lr * cfg.policy_trainer.lr / cfg.wm_trainer.lr)
            t0 = time.perf_counter()
            m_wm = self.wm_trainer.update(self.buffer)
            self._synchronize()
            timers["wm"] += time.perf_counter() - t0
            for k, v in m_wm.items():
                agg[f"wm/{k}"] = agg.get(f"wm/{k}", 0.0) + v
            t0 = time.perf_counter()
            for _ in range(sch.policy_updates_per_wm_update):
                m_pi = self.policy_trainer.update(self.buffer)
                self.num_updates += 1
                for k, v in m_pi.items():
                    agg[f"policy/{k}"] = agg.get(f"policy/{k}", 0.0) + v
            self._synchronize()
            timers["policy"] += time.perf_counter() - t0
            agg["lr"] = lr
            agg_n += 1
            self.iteration += 1
            self._train_seconds = self._time_offset + time.time() - t_start

            # ---------------- logging / checkpoints
            if self.iteration % sch.log_interval == 0:
                it = self.iteration
                elapsed = self._train_seconds
                finished_returns = self._finished_returns
                finished_lens = self._finished_lens
                if self.world > 1:
                    # Every rank must issue the same collectives: reduce (sum, count) unconditionally.
                    tot = allreduce_mean(float(sum(finished_returns))) * self.world
                    cnt = allreduce_mean(float(len(finished_returns))) * self.world
                    length_sum = allreduce_mean(float(sum(finished_lens))) * self.world
                    finished_returns = [tot / cnt] if cnt > 0 else []
                    finished_lens = [length_sum / cnt] if cnt > 0 else []
                    n_finished = round(cnt)
                else:
                    n_finished = len(finished_returns)
                if not self.is_main:
                    agg, agg_n = {}, 0
                    finished_returns, finished_lens = [], []
                    timers = {k: 0.0 for k in timers}
                elif agg_n > 0:
                    for k, v in agg.items():
                        denominator = agg_n * (
                            sch.policy_updates_per_wm_update if k.startswith("policy/") else 1
                        )
                        self.writer.add_scalar(k, v if k == "lr" else v / denominator, it)
                if not self.is_main:
                    tr = float("nan")
                elif finished_returns:
                    tr = sum(finished_returns) / len(finished_returns)
                    self.writer.add_scalar("train/episode_norm_return", tr, it)
                    self.writer.add_scalar(
                        "train/episode_len", sum(finished_lens) / max(len(finished_lens), 1), it
                    )
                else:
                    tr = float("nan")
                with torch.no_grad():
                    if self.is_main and self.buffer.size > 2048:
                        err = self.wm_trainer.online_error(self.buffer, n=1024, k=8)
                        self.writer.add_scalar("wm/online_nse_1step", float(err[0]), it)
                        self.writer.add_scalar("wm/online_nse_8step", float(err[-1]), it)
                    else:
                        err = None
                if self.is_main:
                    for k, v in timers.items():
                        self.writer.add_scalar(f"time/{k}_s_per_iter", v / sch.log_interval, it)
                    self.writer.add_scalar("time/hours", elapsed / 3600, it)
                    self.writer.add_scalar("time/session_hours", (time.time() - t_start) / 3600, it)
                    self.writer.add_scalar("time/env_steps", self.env_steps, it)
                    self.writer.add_scalar("time/updates", self.num_updates, it)
                    self.writer.add_scalar(
                        "sampler/entropy", float(self.command.metrics["sampling_entropy"][0]), it
                    )
                    self.writer.add_scalar(
                        "sampler/bin_return_mean", float(self.command.bin_return.mean()), it
                    )
                    dt_log = time.time() - last_log
                    last_log = time.time()
                    msg = (
                        f"it {it} (update) | {elapsed / 3600:.2f} h | {sch.log_interval / dt_log:.1f} updates/s | env_steps {self.env_steps} | policy_updates {self.num_updates} | "
                        f"train_ret {tr:.3f} (n={n_finished}) | "
                        f"wm nse1 {agg.get('wm/nse_1step', float('nan')) / max(agg_n, 1):.4f} nse8 {(float(err[-1]) if err is not None else float('nan')):.4f} | "
                        f"imag_ret {agg.get('policy/imag_return_undisc', float('nan')) / max(agg_n * sch.policy_updates_per_wm_update, 1):.3f} surv {agg.get('policy/imag_survival_H', float('nan')) / max(agg_n * sch.policy_updates_per_wm_update, 1):.3f} | "
                        f"t sim/wm/pi {timers['sim']:.1f}/{timers['wm']:.1f}/{timers['policy']:.1f}s"
                    )
                    print(msg, flush=True)
                agg, agg_n = {}, 0
                self._finished_returns, self._finished_lens = [], []
                timers = {k: 0.0 for k in timers}
            if self.iteration % sch.checkpoint_interval == 0:
                self.save("ckpt_latest.pt")
                self.save(f"ckpt_{self.iteration:07d}.pt")
            if sch.max_hours is not None and self.iteration % sch.log_interval == 0:
                stop = float((time.time() - t_start) / 3600 > sch.max_hours)
                if self.world > 1:
                    stop = allreduce_mean(
                        stop
                    )  # rank 0's clock decides (all ranks agree on the mean > 0)
                if stop > 0:
                    if self.is_main:
                        print("[INFO] max_hours reached, stopping")
                    break
        self.save("ckpt_latest.pt")
        if self.writer is not None:
            self.writer.flush()
        finalize()
