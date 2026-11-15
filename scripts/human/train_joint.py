#!/usr/bin/env python3
"""Independent Casbot02 motion + human joint training (MorphMimic copy recipe)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--log-root", type=Path)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--init-checkpoint", type=Path)
    source.add_argument("--from-scratch", action="store_true")
    source.add_argument("--resume", type=Path)
    parser.add_argument("--init-bundle", type=Path)
    parser.add_argument("--init-human", type=Path)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--motion-batch", type=int)
    parser.add_argument("--human-batch", type=int)
    parser.add_argument("--wm-batch", type=int)
    parser.add_argument("--wm-batches", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--human-fraction", type=float)
    parser.add_argument("--masking", choices=["mixed", "full", "five"])
    parser.add_argument("--cache-clips", type=int)
    parser.add_argument("--validation-fraction", type=float,
                        help="Held-out clip fraction; new runs default to 0 (train on all pairs)")
    parser.add_argument("--validate-every", type=int)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--replay-capacity", type=int)
    parser.add_argument("--library-device", choices=["cpu", "device"])
    parser.add_argument("--library-transfer", choices=["sync", "pinned"])
    parser.add_argument("--compile", choices=["none", "default", "reduce-overhead"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("cpu-threads must be positive")
    if args.resume and args.init_human:
        parser.error("--resume restores its own human state; do not use --init-human")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = str(args.cpu_threads)
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/supermimic-human-mpl")
    import torch
    from supermimic.config import TrainConfig
    from supermimic.human.data import read_pairs, split_pairs
    from supermimic.human.joint_training import JointTrainer, restore_config, robot_contract, check_warmstart_contract, require_prepared_pairs
    from supermimic.teleop.reference import active_sim_bundle, load_contract
    root = Path(__file__).resolve().parents[2]
    if not args.from_scratch and args.resume is None and args.init_checkpoint is None:
        args.init_checkpoint = root/"logs/casbot02_bones_seed_proprio_cpu7/ckpt_0096000.pt"
    path = args.resume or args.init_checkpoint
    ck = torch.load(path, map_location="cpu", weights_only=False) if path else None
    cfg = restore_config(ck["config"]) if ck else TrainConfig(robot="casbot02")
    saved = ck.get("joint_config", {}) if args.resume else {}
    def option(key, default):
        value = getattr(args, key)
        return saved.get(key, default) if value is None else value
    cfg.robot = "casbot02"
    cfg.device, cfg.cpu_threads = args.device, args.cpu_threads
    cfg.run_name = args.run_name or (Path(args.resume).resolve().parent.name if args.resume else "casbot02_seed_motion_human")
    cfg.log_root = args.log_root or (Path(args.resume).resolve().parent.parent if args.resume else Path("logs"))
    cfg.resume = args.resume
    if args.num_envs is not None or not args.resume:
        cfg.env.num_envs = option("num_envs", 4096)
    cfg.env.anchor_obs, cfg.env.anchor_ori_obs = False, True
    cfg.env.compact_library = True
    for attr, arg, default in (("library_device", "library_device", "cpu"),
                              ("library_transfer", "library_transfer", "pinned"),
                              ("replay_capacity", "replay_capacity", 1000)):
        if getattr(args, arg) is not None or not args.resume:
            setattr(cfg.env, attr, option(arg, default))
    for obj, attr, arg, default in ((cfg.policy_trainer,"batch_size","motion_batch",2048),
            (cfg.policy_trainer,"horizon","horizon",32),
            (cfg.wm_trainer,"batch_size","wm_batch",2048),
            (cfg.wm_trainer,"batches_per_update","wm_batches",10),
            (cfg.schedule,"total_iterations","iterations",60000),
            (cfg.schedule,"log_interval","log_every",50),
            (cfg.schedule,"checkpoint_interval","save_every",4000)):
        if getattr(args,arg) is not None or not args.resume:
            setattr(obj, attr, option(arg, default))
    if args.compile is not None or not args.resume:
        mode = option("compile", "none")
        cfg.wm_trainer.compile = cfg.policy_trainer.compile = None if mode == "none" else mode
    pairs = Path(option("pairs", "data/human/seed_uniform20/pairs.csv"))
    human = dict(human_batch=option("human_batch",512), human_fraction=option("human_fraction",.5),
        masking=option("masking","mixed"), cache_clips=option("cache_clips",0),
        validation_fraction=option("validation_fraction",0.0), validate_every=option("validate_every",500))
    bundle = args.init_bundle or active_sim_bundle(root)
    if args.check_only:
        require_prepared_pairs(pairs)
        meta, _ = robot_contract()
        if args.init_checkpoint:
            from supermimic.human.teacher import digest
            exported = load_contract(bundle)
            if digest(args.init_checkpoint) != exported["checkpoint_sha256"]:
                raise ValueError("Initialization checkpoint/bundle mismatch")
            check_warmstart_contract(meta, exported)
        train, val = split_pairs(read_pairs(pairs), human["validation_fraction"])
        print(json.dumps({"task":"motion-human-joint-copy", "train_pairs":len(train),
            "validation_pairs":len(val), "obs":930,"motion_goal":690,"human_goal":2254,"actions":27,
            "num_envs":cfg.env.num_envs,"motors_match_initialization":True,
            "original_sm_train_modified":False}, indent=2))
        return
    if cfg.device != "cuda:0":
        parser.error("Use CUDA_VISIBLE_DEVICES to select one GPU; mjlab and this task use local cuda:0")
    torch.cuda.set_device(0)
    trainer = JointTrainer(cfg, pairs=pairs, **human, init_checkpoint=args.init_checkpoint,
                           init_bundle=bundle, init_human=args.init_human)
    trainer.run()


if __name__ == "__main__":
    main()
