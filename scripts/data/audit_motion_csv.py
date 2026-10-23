"""Audit input CSVs against a robot model without changing motion values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from supermimic.data.motion_convert import load_csv
from supermimic.robots import get_robot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("g1", "casbot02"), required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--input-fps", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    profile = get_robot(args.robot)
    model = profile.model()
    names = [model.joint(i).name for i in range(1, model.njnt)]
    files = sorted(args.input_dir.expanduser().glob("*.csv"))
    if not files:
        parser.error("No CSV files found")
    motors = profile.motor_parameters()["motors"] if args.robot == "casbot02" else None
    limits = np.array([motors[name]["velocity_limit"] for name in names]) if motors else None
    clips = []
    for path in files:
        raw = load_csv(path, profile.num_joints)
        excess = np.maximum(
            model.jnt_range[1:, 0] - raw[:, 7:], raw[:, 7:] - model.jnt_range[1:, 1]
        )
        bad = excess > 1e-6
        speed = np.abs(np.diff(raw[:, 7:], axis=0)) * args.input_fps
        root_speed = np.linalg.norm(np.diff(raw[:, :3], axis=0), axis=-1) * args.input_fps
        clips.append(
            {
                "name": path.stem,
                "source_frames": len(raw),
                "frames_outside_joint_limits": int(bad.any(axis=-1).sum()),
                "max_joint_limit_excess_rad": float(max(0, excess.max())),
                "joint_limit_excess_by_name": {
                    name: float(excess[:, i].max())
                    for i, name in enumerate(names)
                    if bad[:, i].any()
                },
                "frames_above_motor_velocity_metadata": int((speed > limits).any(axis=-1).sum())
                if limits is not None
                else None,
                "first_root_speed_m_s": float(root_speed[0]),
                "root_speed_m_s_p99": float(np.quantile(root_speed, 0.99)),
                "max_root_speed_m_s": float(root_speed.max()),
            }
        )
    report = {
        "robot": args.robot,
        "input_fps": args.input_fps,
        "csv_convention": "xyz metres, quaternion xyzw, hinge radians in MJCF joint order",
        "policy": "preserve all source frames and values; no automatic clipping or trimming",
        "joint_names": names,
        "summary": {
            "clips": len(clips),
            "source_frames": sum(c["source_frames"] for c in clips),
            "frames_outside_joint_limits": sum(c["frames_outside_joint_limits"] for c in clips),
            "max_joint_limit_excess_rad": max(c["max_joint_limit_excess_rad"] for c in clips),
            "clips_first_root_speed_above_5_m_s": sum(c["first_root_speed_m_s"] > 5 for c in clips),
        },
        "clips": clips,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    print("Saved", args.out)


if __name__ == "__main__":
    main()
