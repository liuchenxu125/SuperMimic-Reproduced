"""Validate temporary Casbot02 Warp capacities without changing training defaults.

Examples: --mode audit / rollout / train --nconmax 96 --njmax 320 --out report.json
All physics substeps and post-reset forward calls are monitored, including silent
overflow bits. Solver iteration-limit bits are reported separately from capacity.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from unittest.mock import patch

import torch
import warp as wp
from mjlab.envs import ManagerBasedRlEnv
from mujoco_warp import OverflowType

from supermimic.envs import SuperMimicEnvCfgArgs, make_env_cfg
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.state_io import StateIO
from supermimic.eval.benchmark import load_policy_from_checkpoint
from supermimic.training.precision import configure_precision


def process_memory_mib():
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    return sum(
        int(row.split(",")[1].strip())
        for row in rows
        if row.split(",")[0].strip() == str(os.getpid())
    )


class CapacityMonitor:
    def __init__(self, env):
        self.env = env
        self.data = env.sim.wp_data
        d = self.data
        self.nefc, self.nacon, self.ncollision, self.overflow = [
            wp.to_torch(getattr(d, k)) for k in ("nefc", "nacon", "ncollision", "overflow")
        ]
        self.bits = [bit for bit in OverflowType if bit.value and bit.name != "ALL"]
        self.bit_values = torch.tensor([int(bit) for bit in self.bits], device=env.device)
        self.flags = torch.zeros(d.nworld, dtype=torch.int32, device=env.device)
        self.bit_world_samples = torch.zeros(len(self.bits), dtype=torch.int64, device=env.device)
        self.peaks = torch.zeros(3, dtype=torch.int64, device=env.device)
        self.capacity_exceed_samples = torch.zeros(3, dtype=torch.int64, device=env.device)
        self.calls = {"step": 0, "forward": 0}
        self.stream = torch.cuda.ExternalStream(wp.get_stream(env.sim.wp_device).cuda_stream)
        torch.cuda.synchronize()

    def sample(self, kind):
        self.calls[kind] += 1
        with torch.cuda.stream(self.stream):
            values = torch.stack([self.nacon[0], self.ncollision[0], self.nefc.max()]).long()
            torch.maximum(self.peaks, values, out=self.peaks)
            self.flags.bitwise_or_(self.overflow)
            self.bit_world_samples += ((self.overflow[:, None] & self.bit_values) != 0).sum(0)
            self.capacity_exceed_samples += torch.stack(
                [
                    (self.nacon[0] > self.data.naconmax).long(),
                    (self.ncollision[0] > self.data.naconmax).long(),
                    (self.nefc > self.data.njmax).sum(),
                ]
            )

    def install(self):
        for kind in ("step", "forward"):
            original = getattr(self.env.sim, kind)

            def wrapper(_original=original, _kind=kind):
                result = _original()
                self.sample(_kind)
                return result

            setattr(self.env.sim, kind, wrapper)

    def report(self):
        torch.cuda.synchronize()
        peaks = self.peaks.cpu().tolist()
        counts = self.capacity_exceed_samples.cpu().tolist()
        flags = self.flags.cpu()
        capacity_names = {"NEFC", "NJMAX_NNZ", "BROADPHASE", "NARROWPHASE"}
        flag_counts = {bit.name: int(((flags & int(bit)) != 0).sum()) for bit in self.bits}
        return {
            "nworld": self.data.nworld,
            "naconmax_shared": self.data.naconmax,
            "njmax_per_world": self.data.njmax,
            "njmax_pad": self.data.njmax_pad,
            "jacobian_sparse": bool(self.env.sim.wp_model.is_sparse),
            "calls": self.calls,
            "max_contacts_shared": peaks[0],
            "max_broadphase_pairs_shared": peaks[1],
            "max_constraints_per_world": peaks[2],
            "contact_pool_peak_fraction": peaks[0] / self.data.naconmax,
            "broadphase_pool_peak_fraction": peaks[1] / self.data.naconmax,
            "constraint_peak_fraction": peaks[2] / self.data.njmax,
            "raw_exceed_samples": dict(
                zip(["contact_pool", "broadphase_pool", "constraints"], counts)
            ),
            "overflow_worlds_by_bit": flag_counts,
            "overflow_world_samples_by_bit": dict(
                zip([bit.name for bit in self.bits], self.bit_world_samples.cpu().tolist())
            ),
            "capacity_passed": not any(counts)
            and not any(count for name, count in flag_counts.items() if name in capacity_names),
        }


def factory(args):
    def build(env_args, device="cuda:0", *, tf32=True):
        configure_precision(tf32)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        cfg = make_env_cfg(env_args)
        cfg.sim.nconmax, cfg.sim.njmax = args.nconmax, args.njmax
        return ManagerBasedRlEnv(cfg, device=device)

    return build


def audit(env, monitor):
    lib = env.command_manager.get_term("motion").library
    io = StateIO(env)
    ids = torch.arange(env.num_envs, device=env.device)
    max_nefc = -1
    worst = None
    max_contacts_world = 0
    histogram = torch.zeros(2048, dtype=torch.int64, device=env.device)
    for offset in range(0, lib.num_frames, env.num_envs):
        frames = (ids + offset).clamp_max(lib.num_frames - 1)
        qpos, qvel = lib.ref_qpos_qvel(frames)
        io.write(ids, torch.cat([qpos, qvel], -1))
        torch.cuda.synchronize()
        valid = min(env.num_envs, lib.num_frames - offset)
        nefc = monitor.nefc[:valid]
        histogram += torch.bincount(nefc.long().clamp_max(2047), minlength=2048)
        peak, index = nefc.max(0)
        peak, index = int(peak), int(index)
        if peak > max_nefc:
            max_nefc = peak
            frame = offset + index
            clip = next(c for c in lib.clips if c.start <= frame < c.start + c.length)
            worst = {"clip": clip.name, "local_frame": frame - clip.start, "constraints": peak}
        ncon = min(int(monitor.nacon[0]), monitor.data.naconmax)
        worldids = wp.to_torch(monitor.data.contact.worldid)[:ncon]
        max_contacts_world = max(
            max_contacts_world, int(torch.bincount(worldids.long(), minlength=env.num_envs).max())
        )
        if offset // env.num_envs % 25 == 0:
            print(
                f"[AUDIT] {min(offset + env.num_envs, lib.num_frames)}/{lib.num_frames}; "
                f"max nefc={max_nefc}, max contacts/world={max_contacts_world}",
                flush=True,
            )
    return {
        "clips": lib.num_clips,
        "frames": lib.num_frames,
        "worst_reference": worst,
        "max_contacts_one_world": max_contacts_world,
        "constraint_histogram": {str(i): int(n) for i, n in enumerate(histogram.cpu()) if n},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["audit", "rollout", "train"], required=True)
    p.add_argument("--nconmax", type=int, default=96)
    p.add_argument("--njmax", type=int, default=320)
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--npz-dir", type=Path, default=Path("data/npz/lafan1_casbot02_codex"))
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument(
        "--trim-torch-cache",
        action="store_true",
        help="Diagnostic only: release unused Torch cache before collection after an update",
    )
    p.add_argument("--action-mode", choices=["policy", "random", "zero"], default="policy")
    p.add_argument("--ckpt", type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if (
        min(args.nconmax, args.njmax, args.num_envs, args.steps, args.iterations, args.batch_size)
        <= 0
    ):
        p.error("capacities and run lengths must be positive")
    if args.out.exists():
        p.error(f"Output exists: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    configure_precision(True)
    started = time.perf_counter()
    report = {
        "args": vars(args).copy(),
        "gpu": torch.cuda.get_device_name(),
        "purpose": "temporary capacity validation; no production defaults changed",
    }
    env = trainer = monitor = None
    try:
        if args.mode == "train":
            from supermimic.config import TrainConfig
            from supermimic.training.loop import Trainer

            cfg = TrainConfig(robot="casbot02", run_name=args.out.stem, seed=args.seed)
            cfg.data.npz_dir = args.npz_dir
            cfg.env.num_envs = args.num_envs
            cfg.policy_trainer.batch_size = args.batch_size
            cfg.schedule.total_iterations = args.iterations
            cfg.schedule.checkpoint_interval = args.iterations
            cfg.schedule.log_interval = min(50, args.iterations)
            with patch("supermimic.training.loop.make_env", factory(args)):
                trainer = Trainer(cfg)
            env = trainer.env
        else:
            files = discover_npz(args.npz_dir)
            training = args.mode == "rollout"
            env = factory(args)(
                SuperMimicEnvCfgArgs(
                    npz_files=[str(f) for f in files],
                    robot="casbot02",
                    num_envs=args.num_envs,
                    obs_noise=training,
                    reset_noise=training,
                    domain_randomization=training,
                    action_noise_std=0.1 if training else 0.0,
                    record_transitions=False,
                    seed=args.seed,
                )
            )
        monitor = CapacityMonitor(env)
        monitor.install()
        cache_trims = {"count": 0, "released_reserved_mib": 0.0, "seconds": 0.0}
        if trainer is not None and args.trim_torch_cache:
            original_collect = trainer._collect_step
            last_trim_update = 0

            def collect_with_cache_trim():
                nonlocal last_trim_update
                if trainer.iteration > last_trim_update:
                    before = torch.cuda.memory_reserved()
                    t0 = time.perf_counter()
                    torch.cuda.empty_cache()
                    cache_trims["seconds"] += time.perf_counter() - t0
                    cache_trims["count"] += 1
                    cache_trims["released_reserved_mib"] += (
                        before - torch.cuda.memory_reserved()
                    ) / 2**20
                    last_trim_update = trainer.iteration
                original_collect()

            trainer._collect_step = collect_with_cache_trim
        report["cache_trims"] = cache_trims
        if trainer is not None:
            trainer.run()
            report["training"] = {
                "iterations": trainer.iteration,
                "env_steps": trainer.env_steps,
                "batch_size": args.batch_size,
                "precision": trainer.precision,
                "all_weights_finite": all(
                    torch.isfinite(p).all().item()
                    for m in (trainer.wm, trainer.policy, trainer.critic)
                    for p in m.parameters()
                ),
            }
            if not report["training"]["all_weights_finite"]:
                raise FloatingPointError("Nonfinite model parameters after training")
        else:
            obs, _ = env.reset()
            if args.mode == "audit":
                report["audit"] = audit(env, monitor)
            else:
                policy = None
                if args.action_mode == "policy":
                    if args.ckpt is None:
                        p.error("policy rollout needs --ckpt")
                    policy, cfg = load_policy_from_checkpoint(args.ckpt, str(env.device))
                    if cfg.get("robot") != "casbot02" or cfg["policy"]["goal_dim"] != 640:
                        raise ValueError("Use a Casbot02 proprio checkpoint")
                n_done = 0
                returns = 0.0
                for step in range(args.steps):
                    with torch.no_grad():
                        action = (
                            policy(obs["policy"], obs["goal"])
                            if policy is not None
                            else (
                                torch.randn(env.num_envs, 27, device=env.device)
                                if args.action_mode == "random"
                                else torch.zeros(env.num_envs, 27, device=env.device)
                            )
                        )
                        obs, reward, term, tout, _ = env.step(action)
                    if not torch.isfinite(reward).all() or not torch.isfinite(obs["policy"]).all():
                        raise FloatingPointError("Nonfinite rollout observation/reward")
                    n_done += int((term | tout).sum())
                    returns += float(reward.mean())
                    if (step + 1) % 250 == 0:
                        print(
                            f"[ROLLOUT] {step + 1}/{args.steps}; resets={n_done}; "
                            f"max nefc={monitor.report()['max_constraints_per_world']}",
                            flush=True,
                        )
                report["rollout"] = {
                    "steps": args.steps,
                    "resets": n_done,
                    "mean_step_reward": returns / args.steps,
                }
        report["passed"] = monitor.report()["capacity_passed"]
    except Exception as error:
        report["passed"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if trainer is not None:
            report["training_progress"] = {
                "outer_updates": trainer.iteration,
                "policy_updates": trainer.num_updates,
                "env_steps": trainer.env_steps,
            }
        if monitor is not None:
            report["capacity"] = monitor.report()
        report["wall_seconds"] = time.perf_counter() - started
        report["process_gpu_memory_mib"] = process_memory_mib()
        report["torch_allocated_mib"] = torch.cuda.memory_allocated() / 2**20
        report["torch_reserved_mib"] = torch.cuda.memory_reserved() / 2**20
        report["torch_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        report["torch_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
        args.out.write_text(json.dumps(report, indent=2, default=str) + "\n")
        print("[REPORT]", args.out, "passed=", report.get("passed"), flush=True)
        if trainer is not None:
            trainer.writer.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
