"""Train a NEW Casbot02 walk4 policy using only IMU and joint observations.

Example: env -u PYTHONPATH uv run python scripts/train_casbot02_proprio.py
Use --print-config to inspect the preset without starting GPU training.
"""

import json
from dataclasses import asdict
from pathlib import Path

import torch
import tyro

from supermimic.config import TrainConfig


def make_config(
    run_name="casbot02_walk4_proprio",
    max_hours=1.0,
    num_envs=4096,
    policy_batch_size=2048,
    resume=None,
):
    cfg = TrainConfig(robot="casbot02", run_name=run_name)
    cfg.data.npz_dir = Path("data/npz/lafan1_casbot02")
    cfg.data.clips = ["walk4_subject1"]
    cfg.env.anchor_obs = False
    cfg.env.num_envs = num_envs
    cfg.policy.num_joints = 27
    cfg.policy.goal_dim = 640
    cfg.policy_trainer.anchor_obs = False
    cfg.policy_trainer.batch_size = policy_batch_size
    spec = cfg.world_model.state_spec
    spec.num_joints = 27
    spec.xi_offset = (0.75,) + (0.0,) * 30
    spec.xi_scale = (0.45,) + (0.01,) * 27 + (0.05,) * 3
    cfg.schedule.max_hours = max_hours
    cfg.resume = resume
    return cfg


def main(
    run_name: str = "casbot02_walk4_proprio",
    max_hours: float = 1.0,
    num_envs: int = 4096,
    policy_batch_size: int = 2048,
    resume: Path | None = None,
    print_config: bool = False,
):
    if min(max_hours, num_envs, policy_batch_size) <= 0:
        raise ValueError("Training duration and batch sizes must be positive")
    cfg = make_config(run_name, max_hours, num_envs, policy_batch_size, resume)
    if print_config:
        print(json.dumps(asdict(cfg), indent=2, default=str))
        return
    if resume is not None:
        saved = torch.load(resume, map_location="cpu", weights_only=False)["config"]
        if (
            saved.get("robot") != "casbot02"
            or saved["env"].get("anchor_obs", False)
            or saved["policy"]["goal_dim"] != 640
            or saved["data"]["clips"] != ["walk4_subject1"]
        ):
            raise ValueError(
                "Resume requires a Casbot02 640-goal walk4 checkpoint. "
                "The existing 649-goal checkpoint cannot be resumed as this policy."
            )
    elif (cfg.log_root / run_name / "config.json").exists():
        raise FileExistsError("Run already exists; use --resume or a different --run-name")
    from supermimic.training.loop import Trainer

    Trainer(cfg).run()


if __name__ == "__main__":
    tyro.cli(main)
