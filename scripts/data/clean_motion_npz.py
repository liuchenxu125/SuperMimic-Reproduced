"""Trim diagnosed clip starts, clamp joints, and rebuild all dependent motion fields.

Only the explicitly selected input directory is replaced. Originals are backed
up outside it, and all outputs are staged and validated before replacement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import mujoco
import numpy as np

from supermimic.data.motion_convert import (
    ResampledMotion,
    _so3_derivative,
    forward_kinematics,
    quat_rotate_inverse,
)
from supermimic.robots import get_robot

TIME_KEYS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def float32_bounds(limits):
    """Float32 bounds strictly INSIDE float64 model limits, including rounding."""
    lo, hi = limits[:, 0].astype(np.float32), limits[:, 1].astype(np.float32)
    lo = np.where(lo.astype(np.float64) < limits[:, 0], np.nextafter(lo, np.float32(np.inf)), lo)
    hi = np.where(hi.astype(np.float64) > limits[:, 1], np.nextafter(hi, np.float32(-np.inf)), hi)
    return lo, hi


def limit_statistics(q, limits):
    excess = np.maximum(limits[:, 0] - q, q - limits[:, 1])
    bad = excess > 1e-6  # Same tolerance as the original CSV audit.
    return {
        "frames": len(q),
        "frames_outside_hard_limits": int(bad.any(axis=1).sum()),
        "joint_values_outside_hard_limits": int(bad.sum()),
        "max_excess_rad": float(max(0.0, excess.max())),
    }


def clean_arrays(arrays, model, root_name, trim_frames):
    names = [model.joint(i).name for i in range(1, model.njnt)]
    bodies = [model.body(i).name for i in range(1, model.nbody)]
    if arrays["joint_names"].tolist() != names or arrays["body_names"].tolist() != bodies:
        raise ValueError("Motion joint/body names differ from the robot model")
    frames, n = arrays["joint_pos"].shape
    if n != model.nq - 7 or frames - trim_frames < 3 or trim_frames < 0:
        raise ValueError("Invalid joint/frame count or trim")
    for key in TIME_KEYS:
        if len(arrays[key]) != frames or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Invalid motion field {key}")
    if not np.isfinite(arrays["fps"]).all() or float(arrays["fps"].item()) <= 0:
        raise ValueError("Invalid FPS")
    result = {
        k: (v[trim_frames:].copy() if k in TIME_KEYS else v.copy()) for k, v in arrays.items()
    }
    root = bodies.index(root_name)
    root_pos = result["body_pos_w"][:, root].astype(np.float64)
    root_quat = result["body_quat_w"][:, root].astype(np.float64)
    norms = np.linalg.norm(root_quat, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("Invalid root quaternion")
    root_quat /= norms
    q_original = result["joint_pos"].astype(np.float64)
    limits = model.jnt_range[1:]
    lo, hi = float32_bounds(limits)
    q = np.clip(q_original, lo.astype(np.float64), hi.astype(np.float64)).astype(np.float32)
    dt = 1 / float(result["fps"].item())
    motion = ResampledMotion(
        fps=1 / dt,
        root_pos=root_pos,
        root_quat=root_quat,
        joint_pos=q.astype(np.float64),
        root_lin_vel_w=np.gradient(root_pos, dt, axis=0),
        root_ang_vel_w=_so3_derivative(root_quat, dt),
        joint_vel=np.gradient(q.astype(np.float64), dt, axis=0),
    )
    fk = forward_kinematics(model, motion)
    result.update(
        joint_pos=q,
        joint_vel=motion.joint_vel.astype(np.float32),
        body_pos_w=fk.pos_w.astype(np.float32),
        body_quat_w=fk.quat_w.astype(np.float32),
        body_lin_vel_w=fk.lin_vel_w.astype(np.float32),
        body_ang_vel_w=fk.ang_vel_w.astype(np.float32),
    )
    changes = np.abs(q.astype(np.float64) - q_original)
    stats = {
        "frames_before": frames,
        "trimmed_frames": trim_frames,
        "trimmed_seconds": trim_frames * dt,
        "frames_after": len(q),
        "before": limit_statistics(arrays["joint_pos"].astype(np.float64), limits),
        "retained_before_clamping": limit_statistics(q_original, limits),
        "after": limit_statistics(q.astype(np.float64), limits),
        "frames_with_joint_changes": int((changes > 0).any(axis=1).sum()),
        "joint_values_changed": int((changes > 0).sum()),
        "max_joint_adjustment_rad": float(changes.max()),
        "joint_changes": {
            name: {
                "values_changed": int((changes[:, i] > 0).sum()),
                "max_adjustment_rad": float(changes[:, i].max()),
            }
            for i, name in enumerate(names)
            if (changes[:, i] > 0).any()
        },
        "first_root_speed_before_m_s": float(np.linalg.norm(arrays["body_lin_vel_w"][0, root])),
        "first_root_speed_after_m_s": float(np.linalg.norm(motion.root_lin_vel_w[0])),
    }
    return result, stats


def validate_arrays(arrays, model, root_name):
    q = arrays["joint_pos"].astype(np.float64)
    limits = model.jnt_range[1:]
    if ((q < limits[:, 0]) | (q > limits[:, 1])).any():
        raise ValueError("Post-cleaning joint is outside an exact model hard limit")
    for key in TIME_KEYS:
        if len(arrays[key]) != len(q) or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Invalid saved array {key}")
    dt = 1 / float(arrays["fps"].item())
    np.testing.assert_allclose(
        arrays["joint_vel"], np.gradient(q, dt, axis=0), atol=2e-5, rtol=2e-6
    )
    names = arrays["body_names"].tolist()
    root = names.index(root_name)
    p = arrays["body_pos_w"][:, root].astype(np.float64)
    quat = arrays["body_quat_w"][:, root].astype(np.float64)
    lin = np.gradient(p, dt, axis=0)
    ang = _so3_derivative(quat, dt)
    np.testing.assert_allclose(arrays["body_lin_vel_w"][:, root], lin, atol=2e-5, rtol=2e-6)
    np.testing.assert_allclose(arrays["body_ang_vel_w"][:, root], ang, atol=3e-5, rtol=3e-6)
    data = mujoco.MjData(model)
    velocity = np.zeros(6)
    max_pos, max_vel = 0.0, 0.0
    samples = np.unique(
        [0, 1, 2, len(q) // 4, len(q) // 2, 3 * len(q) // 4, len(q) - 2, len(q) - 1]
    )
    for t in samples:
        data.qpos[:] = np.r_[p[t], quat[t] / np.linalg.norm(quat[t]), q[t]]
        data.qvel[:] = np.r_[lin[t], quat_rotate_inverse(quat[t], ang[t]), arrays["joint_vel"][t]]
        mujoco.mj_forward(model, data)
        max_pos = max(max_pos, float(np.max(np.abs(arrays["body_pos_w"][t] - data.xpos[1:]))))
        np.testing.assert_allclose(arrays["body_pos_w"][t], data.xpos[1:], atol=2e-5, rtol=2e-6)
        dots = np.abs(np.sum(arrays["body_quat_w"][t] * data.xquat[1:], axis=1))
        np.testing.assert_allclose(dots, 1.0, atol=2e-6, rtol=0)
        for b, _ in enumerate(names, start=1):
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, b, velocity, 0)
            stored = np.r_[arrays["body_ang_vel_w"][t, b - 1], arrays["body_lin_vel_w"][t, b - 1]]
            max_vel = max(max_vel, float(np.max(np.abs(stored - velocity))))
            np.testing.assert_allclose(stored, velocity, atol=5e-5, rtol=1e-5)
    return {
        "independent_fk_sampled_frames": len(samples),
        "max_sampled_body_position_error_m": max_pos,
        "max_sampled_body_velocity_error": max_vel,
        "all_frames_joint_and_root_derivatives_checked": True,
    }


def process_clip(job):
    source, output, robot, trim_frames = job
    profile = get_robot(robot)
    model = profile.model()
    with np.load(source, allow_pickle=False) as loaded:
        arrays = {k: loaded[k] for k in loaded.files}
    cleaned, stats = clean_arrays(arrays, model, profile.root_body, trim_frames)
    with Path(output).open("wb") as f:
        np.savez_compressed(f, **cleaned)
    # Validate the actual serialized data, including independent FK/velocities.
    with np.load(output, allow_pickle=False) as loaded:
        stats["validation"] = validate_arrays(loaded, model, profile.root_body)
    stats.update(
        name=Path(source).stem, original_sha256=digest(source), cleaned_sha256=digest(output)
    )
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--robot", default="casbot02", choices=["casbot02", "g1"])
    parser.add_argument("--trim-seconds", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--apply", action="store_true", help="Replace input clips after backup and validation"
    )
    args = parser.parse_args()
    directory = args.input_dir.resolve()
    if (directory / "cleaning_report.json").exists():
        parser.error(
            "This directory already has a cleaning report; refuse to trim it a second time"
        )
    files = sorted(directory.glob("*.npz"))
    if (
        not files
        or args.workers < 1
        or not np.isfinite(args.trim_seconds)
        or args.trim_seconds <= 0
    ):
        parser.error("Need NPZ files, positive worker count and trim duration")
    benchmark = json.loads(args.benchmark.read_text())
    results = {c["name"]: c for c in benchmark["clips"]}
    trimmed = []
    jobs = []
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    report_dir = args.report_dir.resolve()
    if report_dir.exists():
        parser.error("Report directory already exists; choose a new one")
    report_dir.mkdir(parents=True)
    staging = report_dir / "staging"
    staging.mkdir()
    hashes = {p.name: digest(p) for p in files}
    backup = directory.with_name(directory.name + "_backup_" + stamp)
    if args.apply:
        shutil.copytree(directory, backup)
        if any(digest(backup / name) != value for name, value in hashes.items()):
            raise RuntimeError("Backup hashes differ from input")
    for p in files:
        with np.load(p, allow_pickle=False) as loaded:
            fps = float(loaded["fps"].item())
            frames = len(loaded["joint_pos"])
        if p.stem not in results or results[p.stem]["frames"] != frames:
            raise ValueError(f"{p.name}: source frames do not match benchmark")
        c = results[p.stem]
        trim = (
            round(args.trim_seconds * fps) if c["failed"] and 0 <= c["fail_step"] <= fps else 0
        )
        if trim and not np.isclose(trim / fps, args.trim_seconds):
            raise ValueError("Trim duration does not map to an exact number of frames")
        if trim:
            trimmed.append(
                {
                    "name": p.stem,
                    "original_fail_step": c["fail_step"],
                    "original_fail_seconds": c["fail_step"] / fps,
                    "trim_frames": trim,
                }
            )
        jobs.append((backup / p.name if args.apply else p, staging / p.name, args.robot, trim))
    stats = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_clip, job) for job in jobs]
        for future in as_completed(futures):
            item = future.result()
            stats.append(item)
            print(
                f"[{len(stats)}/{len(files)}] {item['name']}: trim={item['trimmed_frames']}, "
                f"changed={item['joint_values_changed']}, limits/FK/velocities verified",
                flush=True,
            )
    stats.sort(key=lambda c: c["name"])
    if any(digest(directory / name) != value for name, value in hashes.items()):
        raise RuntimeError("Input changed during processing; nothing will be replaced")
    report = {
        "format": "supermimic-motion-cleaning-v1",
        "applied": False,
        "input_dir": str(directory),
        "backup_dir": str(backup) if args.apply else None,
        "benchmark": str(args.benchmark.resolve()),
        "benchmark_sha256": digest(args.benchmark),
        "robot": args.robot,
        "trim_seconds": args.trim_seconds,
        "trimmed_clips": trimmed,
        "limits_policy": "MuJoCo model hard limits; float32 bounds rounded inward",
        "derivative_policy": "Recompute joint/root linear differences and SO(3) angular differences after trim/clamp; rebuild every body's FK and spatial velocity",
        "source_audit_note": "source_audit.json still describes the ORIGINAL 30 FPS CSVs; cleaning statistics below describe the copied 50 FPS NPZs",
        "summary": {
            "clips": len(stats),
            "trimmed_clips": len(trimmed),
            "frames_before": sum(c["frames_before"] for c in stats),
            "frames_removed": sum(c["trimmed_frames"] for c in stats),
            "frames_after": sum(c["frames_after"] for c in stats),
            "frames_outside_hard_limits_before": sum(
                c["before"]["frames_outside_hard_limits"] for c in stats
            ),
            "frames_outside_hard_limits_after": sum(
                c["after"]["frames_outside_hard_limits"] for c in stats
            ),
            "joint_values_changed": sum(c["joint_values_changed"] for c in stats),
            "max_joint_adjustment_rad": max(c["max_joint_adjustment_rad"] for c in stats),
        },
        "clips": stats,
    }
    (report_dir / "cleaning_report.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.apply:
        for p in files:
            os.chmod(staging / p.name, (directory / p.name).stat().st_mode & 0o777)
            os.replace(staging / p.name, directory / p.name)
        if any(digest(directory / (c["name"] + ".npz")) != c["cleaned_sha256"] for c in stats):
            raise RuntimeError("Post-replacement checksum mismatch; restore from backup")
        report["applied"] = True
        content = json.dumps(report, indent=2) + "\n"
        (directory / "cleaning_report.json").write_text(content)
        (report_dir / "cleaning_report.json").write_text(content)
        staging.rmdir()
    print(json.dumps(report["summary"], indent=2))
    print("Backup:", report["backup_dir"])
    print("Report:", report_dir / "cleaning_report.json")


if __name__ == "__main__":
    main()
