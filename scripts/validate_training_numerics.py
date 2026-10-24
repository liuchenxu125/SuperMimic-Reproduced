"""Bounded real Casbot02 training check, keeping the full-run learning-rate schedule.

This is a numerical regression check, not a tracking-performance benchmark.
Runs from scratch, never resumes divergent checkpoints or changes the dataset.
"""

import argparse
import json
from pathlib import Path

import torch

from supermimic.config import TrainConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--npz-dir", type=Path, default=Path("data/npz/lafan1_casbot02"))
    parser.add_argument("--anchor-obs", action="store_true")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    if min(args.iterations, args.num_envs, args.batch_size) <= 0:
        parser.error("iterations, num-envs and batch-size must be positive")
    cfg = TrainConfig(robot="casbot02", run_name=args.run_name)
    cfg.data.npz_dir = args.npz_dir
    cfg.env.anchor_obs = args.anchor_obs
    cfg.env.num_envs = args.num_envs
    cfg.policy_trainer.batch_size = args.batch_size
    # Limit the run without accelerating the original 250,000-step LR decay.
    cfg.schedule.total_iterations = args.iterations
    cfg.schedule.lr_decay_fraction = 250_000 / args.iterations
    cfg.schedule.checkpoint_interval = args.iterations
    output = cfg.log_root / args.run_name
    if output.exists():
        parser.error(f"Run already exists: {output}; choose a new --run-name")
    from supermimic.training.loop import Trainer

    trainer = Trainer(cfg)
    report = {
        "purpose": "numerical stability only; not deployable-policy qualification",
        "anchor_obs": args.anchor_obs,
        "npz_dir": str(args.npz_dir),
        "seed": cfg.seed,
        "num_envs": args.num_envs,
        "policy_batch_size": args.batch_size,
        "requested_iterations": args.iterations,
        "learning_rate_decay_steps": 250_000,
    }
    try:
        trainer.run()
        modules = {"world_model": trainer.wm, "policy": trainer.policy, "critic": trainer.critic}
        report["nonfinite_tensors"] = {
            name: [
                key for key, value in module.state_dict().items() if not torch.isfinite(value).all()
            ]
            for name, module in modules.items()
        }
        if any(report["nonfinite_tensors"].values()):
            raise FloatingPointError("Nonfinite checkpoint tensors after validation")
        report["passed"] = True
    except Exception as error:
        report["passed"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["completed_iterations"] = trainer.iteration
        report["updates"] = trainer.num_updates
        (output / "numerics_validation.json").write_text(json.dumps(report, indent=2) + "\n")
        trainer.writer.close()
        trainer.env.close()
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
