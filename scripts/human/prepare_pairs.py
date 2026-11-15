#!/usr/bin/env python3
"""SOMA Uniform BVH/human50 NPZ -> strict, separate Casbot02 human pair manifest."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, help="Existing robot keep-list; never modified")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--bvh-root", type=Path)
    group.add_argument(
        "--human-root", type=Path, help="Already converted MorphMimic human50 sidecars"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--min-frames", type=int, default=52, help="10 warmup + 32 imagined + 10 preview"
    )
    args = parser.parse_args()
    from supermimic.envs.mdp.motion_library import discover_npz
    from supermimic.human.data import Pair, load_pair
    from supermimic.human.soma import convert_human_clip
    from supermimic.teleop.reference import active_sim_bundle, load_contract

    root = Path(__file__).resolve().parents[2]
    meta = load_contract(args.bundle or active_sim_bundle(root))
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(
            "Use a new empty --output-dir; original sidecars/robot data will not be overwritten"
        )
    if output == args.robot_dir.resolve() or (
        args.human_root and output == args.human_root.resolve()
    ):
        parser.error("Output must be separate from source data")
    source_root = (args.bvh_root or args.human_root).resolve()
    index = defaultdict(list)
    for path in source_root.rglob("*.bvh" if args.bvh_root else "*.npz"):
        index[path.stem].append(path)
    files = discover_npz(args.robot_dir, manifest=args.manifest)
    rows, audit = [], []
    output.mkdir(parents=True, exist_ok=True)
    for file in files:
        file = Path(file).resolve()
        matches = index.get(file.stem, [])
        if not matches:
            audit.append({"name": file.stem, "status": "missing_human"})
            continue
        if args.limit is not None and len(rows) >= args.limit:
            break
        if len(matches) > 1:
            audit.append({"name": file.stem, "status": "ambiguous_human_name"})
            continue
        source = matches[0]
        try:
            if args.bvh_root:
                target = output / "human50" / source.relative_to(source_root).with_suffix(".npz")
                convert_human_clip(str(source), str(target))
            else:
                target = source
            pair = Pair(file.stem, file, target)
            robot, _ = load_pair(pair, meta)
            if len(robot) < args.min_frames:
                raise ValueError(f"too_short: {len(robot)} < {args.min_frames}")
            rows.append(
                {
                    "name": file.stem,
                    "robot_npz": str(file),
                    "human_npz": str(target),
                    "robot_start_frame": 0,
                    "human_start_frame": 0,
                    "length": len(robot),
                }
            )
            audit.append(
                {
                    "name": file.stem,
                    "status": "ready",
                    "frames": len(robot),
                    "human_source": str(source),
                }
            )
        except (ValueError, KeyError, OSError) as exc:
            audit.append({"name": file.stem, "status": "rejected", "reason": str(exc)})
    fields = ["name", "robot_npz", "human_npz", "robot_start_frame", "human_start_frame", "length"]
    with (output / "pairs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {"ready": len(rows), "audit": audit, "robot_data_modified": False}
    (output / "preparation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    print(
        json.dumps(
            {
                "ready": len(rows),
                "manifest": str(output / "pairs.csv"),
                "report": str(output / "preparation_report.json"),
            },
            ensure_ascii=False,
        )
    )
    if not rows:
        raise SystemExit(
            "No strict time pairs; inspect preparation_report.json, do not silently truncate mismatches"
        )


if __name__ == "__main__":
    main()
