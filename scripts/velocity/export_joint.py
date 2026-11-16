#!/usr/bin/env python3
"""Export separate motion/velocity actors sharing the jointly-trained decoder."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from supermimic.policy.nets import Policy, PolicyCfg
from supermimic.velocity.policy import VelocityActor, VelocityEncoder
from supermimic.velocity.runner import FORMAT


class MotionActor(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.encoder, self.decoder = policy.encoders["motion"], policy.decoder

    def forward(self, history, goal):
        return self.decoder(self.encoder(goal), history)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if ck.get("velocity_format") != FORMAT:
        raise ValueError("Expected a motion+velocity checkpoint")
    cfg = PolicyCfg(**ck["config"]["policy"])
    policy = Policy(cfg)
    policy.encoders["velocity"] = VelocityEncoder(cfg)
    policy.load_state_dict(ck["policy"])
    policy.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    proprio_dim = cfg.history * (3 * cfg.num_joints + 6)
    obs_dim = len(policy.decoder.scale)
    for name, actor, width, goal in (
        ("motion", MotionActor(policy), obs_dim, cfg.goal_dim),
        ("velocity", VelocityActor(policy), proprio_dim, 3),
    ):
        actor.eval()
        example = (torch.zeros(2, width), torch.zeros(2, goal))
        script = torch.jit.trace(actor, example, check_trace=False)
        for batch in (1, 4):
            x = (torch.randn(batch, width) * 0.1, torch.randn(batch, goal) * 0.1)
            torch.testing.assert_close(script(*x), actor(*x), rtol=1e-5, atol=1e-6)
        script.save(str(args.output_dir / (name + "_actor.pt")))
    # Inference/evaluation artifact only: omit optimizer state, so a velocity
    # joint optimizer cannot be mistaken for a native motion-only resume.
    native = {
        k: ck[k]
        for k in ("wm", "critic", "config", "iteration", "checkpoint_version", "iteration_unit")
    }
    native["policy"] = {
        k: v for k, v in ck["policy"].items() if not k.startswith("encoders.velocity.")
    }
    torch.save(native, args.output_dir / "motion_eval_checkpoint.pt")
    meta = {
        "format": FORMAT,
        "iteration": ck["iteration"],
        "motion_history_dim": obs_dim,
        "motion_goal_dim": cfg.goal_dim,
        "velocity_history_dim": proprio_dim,
        "velocity_goal_dim": 3,
        "velocity_goal_names": ["vx_body_mps", "vy_body_mps", "wz_body_radps"],
        "actions": cfg.num_joints,
        "velocity_config": ck["velocity_config"],
        "robot_identity": ck["robot_identity"],
        "note": "Motion eval checkpoint is not a training resume checkpoint",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
