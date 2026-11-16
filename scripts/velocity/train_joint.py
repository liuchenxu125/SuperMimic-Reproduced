#!/usr/bin/env python3
"""Independent motion + velocity joint training; does not change sm-train/human."""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="casbot02", choices=["casbot02"])
    parser.add_argument("--run-name")
    parser.add_argument("--log-root", type=Path)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--from-scratch", action="store_true")
    source.add_argument("--init-checkpoint", type=Path)
    source.add_argument("--resume", type=Path)
    parser.add_argument("--npz-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--clips", nargs="+")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument(
        "--batch-size", type=int, help="TOTAL imagination batch, split between motion/velocity"
    )
    parser.add_argument("--wm-batch", type=int)
    parser.add_argument("--wm-batches", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--library-device", choices=["cpu", "device"])
    parser.add_argument("--library-transfer", choices=["pinned", "sync"])
    parser.add_argument("--compile", choices=["none", "default", "reduce-overhead"])
    parser.add_argument("--velocity-fraction", type=float)
    parser.add_argument("--command-window", type=int)
    parser.add_argument("--command-noise", type=float, nargs=3)
    parser.add_argument("--lin-x-range", type=float, nargs=2)
    parser.add_argument("--lin-y-range", type=float, nargs=2)
    parser.add_argument("--ang-z-range", type=float, nargs=2)
    parser.add_argument("--latent-bc-weight", type=float)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--replay-capacity", type=int)
    parser.add_argument("--data-limit", type=int)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(args.cpu_threads)
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/supermimic-velocity-mpl")
    import torch

    from supermimic.config import TrainConfig
    from supermimic.velocity.commands import VelocityTaskCfg
    from supermimic.velocity.runner import FORMAT, VelocityTrainer, dataset_contract, restore_config

    ck = (
        torch.load(args.resume or args.init_checkpoint, map_location="cpu", weights_only=False)
        if (args.resume or args.init_checkpoint)
        else None
    )
    if args.resume and ck.get("velocity_format") != FORMAT:
        parser.error("--resume requires a joint velocity checkpoint")
    cfg = restore_config(ck["config"]) if ck else TrainConfig(robot=args.robot)
    if cfg.robot != args.robot:
        parser.error("Robot must match the checkpoint")
    cfg.device, cfg.cpu_threads = args.device, args.cpu_threads
    cfg.run_name = args.run_name or (
        args.resume.resolve().parent.name if args.resume else "casbot02_motion_velocity"
    )
    cfg.log_root = args.log_root or (
        args.resume.resolve().parent.parent if args.resume else Path("logs")
    )
    cfg.resume = args.resume
    # Motion contract is fixed to the user's current no-position/per-frame-6D
    # variant. The velocity actor internally replaces only its own extra term.
    cfg.env.anchor_obs, cfg.env.anchor_ori_obs = False, True
    cfg.env.compact_library = True
    for obj, attr, arg, default in (
        (cfg.data, "npz_dir", "npz_dir", Path("data/npz/bones_seed_casbot02_3mm")),
        (cfg.data, "clips", "clips", None),
        (cfg.data, "limit", "data_limit", None),
        (cfg.env, "num_envs", "num_envs", 2048),
        (cfg.env, "library_device", "library_device", "cpu"),
        (cfg.env, "library_transfer", "library_transfer", "pinned"),
        (cfg.env, "replay_capacity", "replay_capacity", 1000),
        (cfg.policy_trainer, "batch_size", "batch_size", 2048),
        (cfg.policy_trainer, "horizon", "horizon", 32),
        (cfg.wm_trainer, "batch_size", "wm_batch", 2048),
        (cfg.wm_trainer, "batches_per_update", "wm_batches", 10),
        (cfg.schedule, "total_iterations", "iterations", 320000),
        (cfg.schedule, "log_interval", "log_every", 50),
        (cfg.schedule, "checkpoint_interval", "save_every", 4000),
    ):
        value = getattr(args, arg)
        if value is not None or not args.resume:
            setattr(obj, attr, value if value is not None else default)
    if args.manifest is not None:
        cfg.data.manifest = args.manifest
    elif not args.resume:
        candidate = cfg.data.npz_dir / "manifest_filtered.csv"
        cfg.data.manifest = candidate if candidate.is_file() else None
    if not args.resume:
        cfg.data.library = None
    if args.compile is not None or not args.resume:
        mode = args.compile or "default"
        cfg.wm_trainer.compile = cfg.policy_trainer.compile = None if mode == "none" else mode
    values = ck["velocity_config"].copy() if args.resume else asdict(VelocityTaskCfg())
    for attr, arg in (
        ("velocity_fraction", "velocity_fraction"),
        ("window_frames", "command_window"),
        ("command_noise", "command_noise"),
        ("lin_vel_x_range", "lin_x_range"),
        ("lin_vel_y_range", "lin_y_range"),
        ("ang_vel_z_range", "ang_z_range"),
        ("latent_bc_weight", "latent_bc_weight"),
    ):
        v = getattr(args, arg)
        if v is not None:
            values[attr] = tuple(v) if isinstance(v, list) else v
    velocity = VelocityTaskCfg(**values)
    velocity.validate()
    if args.check_only:
        print(
            json.dumps(
                {
                    "task": FORMAT,
                    "data": dataset_contract(cfg),
                    "num_envs": cfg.env.num_envs,
                    "total_policy_batch": cfg.policy_trainer.batch_size,
                    "motion_history": 930 if args.robot == "casbot02" else 990,
                    "motion_goal": 690 if args.robot == "casbot02" else 730,
                    "velocity_goal": 3,
                    "velocity": asdict(velocity),
                    "initialization": "weights-only" if args.init_checkpoint else "scratch",
                    "original_sm_train_modified": False,
                },
                indent=2,
            )
        )
        return
    if args.device != "cuda:0":
        parser.error("Use CUDA_VISIBLE_DEVICES to select a GPU; this task uses local cuda:0")
    torch.cuda.set_device(0)
    VelocityTrainer(cfg, velocity, args.init_checkpoint).run()


if __name__ == "__main__":
    main()
