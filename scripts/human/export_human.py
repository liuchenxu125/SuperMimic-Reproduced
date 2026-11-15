#!/usr/bin/env python3
"""Export an independent human actor; never installs it into rl_controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from supermimic.human.constants import HUMAN_GOAL_DIM, KP_GOAL_VERSION
from supermimic.human.features import fixed_human_masks
from supermimic.human.model import HumanEncoder
from supermimic.human.teacher import FrozenTeacher, digest


class HumanActor(nn.Module):
    def __init__(self, teacher, human):
        super().__init__()
        self.encoder, self.decoder = human, teacher.policy.decoder

    def forward(
        self,
        obs_hist,
        human_pos,
        human_quat,
        robot_anchor_quat,
        point_mask,
        frame_mask,
        heading_mask,
    ):
        goal = self.encoder.pack(
            human_pos,
            human_quat,
            robot_anchor_quat,
            point_mask,
            frame_mask,
            heading_mask,
            update=False,
        )
        return self.decoder(self.encoder(goal), obs_hist)


def export(path, output, teacher_ckpt=None, bundle=None):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if (
        ck.get("format") != "supermimic-human-copy-v1"
        or ck.get("human_goal_version") != KP_GOAL_VERSION
    ):
        raise ValueError("Expected a v10 human-copy checkpoint")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty export directory")
    teacher = FrozenTeacher(teacher_ckpt or ck["teacher_checkpoint"], bundle or ck["bundle"], "cpu")
    if (teacher.sha256, teacher.contract_sha256, teacher.model_sha256) != (
        ck["teacher_sha256"],
        ck["contract_sha256"],
        ck["model_sha256"],
    ):
        raise ValueError("Human checkpoint teacher/robot contract mismatch")
    human = HumanEncoder(teacher.pcfg)
    human.load_state_dict(ck["human"])
    module = HumanActor(teacher, human).eval()

    def inputs(batch, points):
        mask = fixed_human_masks(batch, points, 10, torch.device("cpu"))
        quat = torch.zeros(batch, 10, 24, 4)
        quat[..., 0] = 1.0
        root = torch.zeros(batch, 4)
        root[..., 0] = 1.0
        return (
            torch.randn(batch, 930) * 0.1,
            torch.randn(batch, 10, 24, 3) * 0.2,
            quat,
            root,
            *mask,
        )

    with torch.no_grad():
        traced = torch.jit.trace(module, inputs(1, "full"), check_trace=False)
        errors = []
        for batch, points in ((1, "full"), (4, "five"), (2, "full")):
            args = inputs(batch, points)
            a, b = module(*args), traced(*args)
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
            errors.append(float((a - b).abs().max()))
    output.mkdir(parents=True, exist_ok=True)
    traced.save(str(output / "human_actor.pt"))
    manifest = {
        "format": "supermimic-human-actor-v1",
        "human_goal_version": KP_GOAL_VERSION,
        "checkpoint": str(Path(path).resolve()),
        "checkpoint_sha256": digest(path),
        "teacher_sha256": teacher.sha256,
        "iteration": ck["iteration"],
        "obs_dim": 930,
        "motion_goal_dim": 690,
        "human_goal_dim": HUMAN_GOAL_DIM,
        "actions": 27,
        "control_dt": 0.02,
        "inputs": [
            "obs_hist[B,930]",
            "human_pos[B,10,24,3]",
            "human_quat_wxyz[B,10,24,4]",
            "robot_torso_quat_wxyz[B,4]",
            "point_mask[B,24] bool",
            "frame_mask[B,10] bool",
            "heading_mask[B] bool",
        ],
        "human_coordinates": "SOMA uniform skeleton; SMPL rest frames; metres; z-up; shared aligned heading",
        "normal_deployment_changed": False,
        "trace_max_abs_error": max(errors),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-ckpt", type=Path)
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            export(args.checkpoint, args.output_dir, args.teacher_ckpt, args.bundle), indent=2
        )
    )
