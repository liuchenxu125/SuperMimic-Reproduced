#!/usr/bin/env python3
"""Stage audited BONES temporal repairs; --apply installs a verified staged plan.

Only flagged NPZs change. Backups use hard links and all replacements use rename,
so original bytes remain available without copying the entire dataset. Never
modify these backup links in place. Original CSVs are read/stat only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np

from supermimic.data.motion_convert import write_motion_npz
from supermimic.data.temporal_cleaning import (
    THRESHOLDS,
    TIME_KEYS,
    clean_spans,
    jump_edges,
    rebuild_span,
)
from supermimic.robots import get_robot

MODEL = None
OPTIONS = None


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def initialize(options):
    global MODEL, OPTIONS
    OPTIONS = options
    MODEL = get_robot("casbot02").model(collision_model="primitive")


def validate_native(arrays):
    """Independently validate sampled FK and link velocities with MuJoCo."""
    data = mujoco.MjData(MODEL)
    names = list(arrays["body_names"])
    root = names.index("base_link")
    body_ids = [MODEL.body(str(name)).id for name in names]
    from supermimic.data.motion_convert import quat_rotate_inverse

    for t in np.unique(np.linspace(0, len(arrays["joint_pos"]) - 1, 5, dtype=int)):
        data.qpos[:3] = arrays["body_pos_w"][t, root]
        data.qpos[3:7] = arrays["body_quat_w"][t, root]
        data.qpos[7:] = arrays["joint_pos"][t]
        data.qvel[:3] = arrays["body_lin_vel_w"][t, root]
        data.qvel[3:6] = quat_rotate_inverse(data.qpos[3:7], arrays["body_ang_vel_w"][t, root])
        data.qvel[6:] = arrays["joint_vel"][t]
        mujoco.mj_forward(MODEL, data)
        np.testing.assert_allclose(
            arrays["body_pos_w"][t], data.xpos[body_ids], atol=3e-6, rtol=2e-6
        )
        for j, body in enumerate(body_ids):
            velocity = np.empty(6)
            mujoco.mj_objectVelocity(MODEL, data, mujoco.mjtObj.mjOBJ_XBODY, body, velocity, 0)
            np.testing.assert_allclose(
                arrays["body_ang_vel_w"][t, j], velocity[:3], atol=2e-5, rtol=3e-6
            )
            np.testing.assert_allclose(
                arrays["body_lin_vel_w"][t, j], velocity[3:], atol=2e-5, rtol=3e-6
            )


def stage_one(audit):
    name = audit["name"]
    source = Path(OPTIONS["dataset"]) / f"{name}.npz"
    sha = digest(source)
    with np.load(source) as f:
        arrays = {k: f[k] for k in f.files}
    edges = jump_edges(arrays)
    if (
        len(edges) != int(audit["jump_count"])
        or len(arrays["joint_pos"]) != int(audit["frames"])
        or int(edges[0]) != int(audit["first_jump_frame"])
        or int(edges[-1]) != int(audit["last_jump_frame"])
    ):
        raise ValueError(f"{name}: data no longer matches the original audit")
    fps = float(arrays["fps"].item())
    spans, method = clean_spans(
        len(arrays["joint_pos"]), edges, fps, math.ceil(OPTIONS["min_seconds"] * fps) + 1
    )
    result = {
        "name": name,
        "sha256": sha,
        "frames": len(arrays["joint_pos"]),
        "fps": fps,
        "method": method,
        "jump_edges": edges.tolist(),
        "outputs": [],
    }
    for i, (start, end) in enumerate(spans, 1):
        new_name = name if len(spans) == 1 else f"{name}__clean_part{i:02d}"
        repaired = rebuild_span(arrays, MODEL, start, end)
        repaired.update(
            temporal_cleaning=np.array("split_v1"),
            temporal_original_name=np.array(name),
            temporal_original_sha256=np.array(sha),
            temporal_source_frame_start=np.array(start, dtype=np.int64),
            temporal_source_frame_end=np.array(end, dtype=np.int64),
        )
        validate_native(repaired)
        output = Path(OPTIONS["work"]) / "staged" / f"{new_name}.npz"
        write_motion_npz(output, repaired)
        # Check the serialized file, not only the in-memory result.
        with np.load(output) as f:
            for k, v in repaired.items():
                np.testing.assert_array_equal(f[k], v)
        root = list(repaired["body_names"]).index("base_link")
        result["outputs"].append(
            {
                "name": new_name,
                "start": start,
                "end": end,
                "frames": end - start,
                "bytes": output.stat().st_size,
                "sha256": digest(output),
                "max_joint_speed": float(abs(repaired["joint_vel"]).max()),
                "max_root_speed": float(
                    np.linalg.norm(repaired["body_lin_vel_w"][:, root], axis=-1).max()
                ),
            }
        )
    return result


def write_manifest(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def stage(args):
    work, dataset = args.work_dir.resolve(), args.dataset.resolve()
    work.mkdir(parents=True, exist_ok=True)
    if (work / "plan.json").exists():
        raise FileExistsError("Plan already staged; inspect it, then use --apply")
    if shutil.disk_usage(dataset).free < 2 * 1024**3:
        raise OSError("Need at least 2 GiB free for staged repairs")
    with args.audit.open() as stream:
        audit = [r for r in csv.DictReader(stream) if int(r["jump_count"])]
    manifest = args.manifest.resolve()
    with manifest.open() as stream:
        rows = list(csv.DictReader(stream))
    selected = {r["name"] for r in rows if r["keep"] == "1"}
    if not audit or not {r["name"] for r in audit} <= selected:
        raise ValueError("Audit must flag nonempty, currently selected motions")
    options = {"dataset": str(dataset), "work": str(work), "min_seconds": args.min_seconds}
    (work / "staged").mkdir(exist_ok=True)
    repairs = []
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=initialize, initargs=(options,)
    ) as pool:
        for i, result in enumerate(pool.map(stage_one, audit, chunksize=8), 1):
            repairs.append(result)
            if i % 200 == 0:
                print(f"[STAGED] {i}/{len(audit)}", flush=True)
    names = [v["name"] for r in repairs for v in r["outputs"]]
    if len(set(names)) != len(names) or set(names) & (selected - {r["name"] for r in audit}):
        raise ValueError("Cleaned motion names collide")
    by_name = {r["name"]: r for r in repairs}
    updated = []
    for row in rows:
        repair = by_name.get(row["name"])
        if repair is None:
            updated.append(row)
            continue
        if not (len(repair["outputs"]) == 1 and repair["outputs"][0]["name"] == row["name"]):
            updated.append(dict(row, keep=0, reason="temporal_replaced_or_too_short"))
        for output in repair["outputs"]:
            updated.append(
                dict(
                    row,
                    name=output["name"],
                    frames=output["frames"],
                    keep=1,
                    reason="",
                    npz_bytes=output["bytes"],
                    original_name=row["name"],
                    source_frame_start=output["start"],
                    source_frame_end=output["end"],
                    temporal_cleaning=repair["method"],
                )
            )
    updated.sort(key=lambda r: r["name"])
    write_manifest(work / "manifest_temporal_cleaned.csv", updated)
    accepted = [r for r in updated if str(r["keep"]) == "1"]
    before_frames = sum(int(r["frames"]) for r in rows if r["keep"] == "1")
    after_frames = sum(int(r["frames"]) for r in accepted)
    plan = {
        "version": 1,
        "applied": False,
        "dataset": str(dataset),
        "manifest": str(manifest),
        "manifest_sha256": digest(manifest),
        "audit": str(args.audit.resolve()),
        "audit_sha256": digest(args.audit),
        "thresholds": THRESHOLDS,
        "thresholds_origin": "conservative review thresholds, not paper filtering rules",
        "min_segment_seconds": args.min_seconds,
        "min_segment_frames_at_50hz": math.ceil(args.min_seconds * 50) + 1,
        "frame_interval": "[start,end), original 50Hz NPZ",
        "before_clips": len(selected),
        "after_clips": len(accepted),
        "before_frames": before_frames,
        "after_frames": after_frames,
        "removed_frames": before_frames - after_frames,
        "before_duration_hours": (before_frames - len(selected)) / 50 / 3600,
        "after_duration_hours": (after_frames - len(accepted)) / 50 / 3600,
        "affected_originals": len(repairs),
        "unchanged_originals": len(selected) - len(repairs),
        "no_retained_segment": sum(not r["outputs"] for r in repairs),
        "repaired_segments": len(names),
        "repairs": repairs,
        "cleaned_manifest_sha256": digest(work / "manifest_temporal_cleaned.csv"),
    }
    # Record unchanged NPZ identity without copying or rewriting their bytes.
    inventory = []
    for name in sorted(selected - by_name.keys()):
        p = dataset / f"{name}.npz"
        s = p.stat()
        inventory.append([name, s.st_size, s.st_mtime_ns, s.st_ino])
    write_json(work / "unchanged_inventory.json", inventory)
    write_json(work / "plan.json", plan)
    print(json.dumps({k: v for k, v in plan.items() if k != "repairs"}, indent=2), flush=True)


def apply(work):
    plan = json.loads((work / "plan.json").read_text())
    if plan["applied"]:
        raise FileExistsError("Plan already applied; refusing to trim twice")
    dataset, manifest = Path(plan["dataset"]), Path(plan["manifest"])
    cleaned_manifest = work / "manifest_temporal_cleaned.csv"
    if digest(cleaned_manifest) != plan["cleaned_manifest_sha256"]:
        raise ValueError("Staged manifest changed")
    backup = work / "backup"
    backup.mkdir(exist_ok=True)
    old_manifest = backup / manifest.name
    if not old_manifest.exists():
        if digest(manifest) != plan["manifest_sha256"]:
            raise ValueError("Original manifest changed")
        shutil.copy2(manifest, old_manifest)
    if digest(old_manifest) != plan["manifest_sha256"]:
        raise ValueError("Backup manifest changed")
    if digest(manifest) not in {plan["manifest_sha256"], plan["cleaned_manifest_sha256"]}:
        raise ValueError("Active manifest was edited after this plan was staged")
    # Complete ALL backups and verify ALL outputs before changing active files.
    with old_manifest.open() as stream:
        rows = {r["name"]: r for r in csv.DictReader(stream)}
    for repair in plan["repairs"]:
        original = dataset / f"{repair['name']}.npz"
        saved = backup / original.name
        if not saved.exists():
            if digest(original) != repair["sha256"]:
                raise ValueError(f"Original changed: {original}")
            os.link(original, saved)
        if digest(saved) != repair["sha256"]:
            raise ValueError(f"Backup changed: {saved}")
        # On an interrupted apply, an original may already be replaced/deleted.
        # Never overwrite unrelated edits made after the backup was created.
        if original.exists():
            allowed = {repair["sha256"]} | {
                o["sha256"] for o in repair["outputs"] if o["name"] == repair["name"]
            }
            if digest(original) not in allowed:
                raise ValueError(f"Active NPZ was edited after staging: {original}")
        row = rows[repair["name"]]
        source = Path(row["source"])
        stat = source.stat()
        if stat.st_size != int(row["source_bytes"]) or stat.st_mtime_ns != int(
            row["source_mtime_ns"]
        ):
            raise ValueError(f"Original CSV changed since preparation: {source}")
        for output in repair["outputs"]:
            staged = work / "staged" / f"{output['name']}.npz"
            target = dataset / staged.name
            if digest(staged if staged.exists() else target) != output["sha256"]:
                raise ValueError(f"Staged output changed: {output['name']}")
            if (
                target.exists()
                and output["name"] != repair["name"]
                and digest(target) != output["sha256"]
            ):
                raise FileExistsError(f"Unrelated output already exists: {target}")
    for repair in plan["repairs"]:
        for output in repair["outputs"]:
            staged = work / "staged" / f"{output['name']}.npz"
            if staged.exists():
                os.replace(staged, dataset / staged.name)
        if not any(o["name"] == repair["name"] for o in repair["outputs"]):
            (dataset / f"{repair['name']}.npz").unlink(missing_ok=True)
    for target in (dataset / "manifest_temporal_cleaned.csv", manifest):
        temporary = target.with_suffix(".csv.tmp")
        shutil.copy2(cleaned_manifest, temporary)
        os.replace(temporary, target)
    # Verify byte-identical replacements, retained original backups and untouched files.
    for repair in plan["repairs"]:
        for output in repair["outputs"]:
            if digest(dataset / f"{output['name']}.npz") != output["sha256"]:
                raise ValueError("Installed output hash mismatch")
    inventory = json.loads((work / "unchanged_inventory.json").read_text())
    for name, size, mtime, inode in inventory:
        s = (dataset / f"{name}.npz").stat()
        if (s.st_size, s.st_mtime_ns, s.st_ino) != (size, mtime, inode):
            raise ValueError(f"Unflagged NPZ changed: {name}")
    with cleaned_manifest.open() as stream:
        expected = {r["name"] for r in csv.DictReader(stream) if r["keep"] == "1"}
    if {p.stem for p in dataset.glob("*.npz")} != expected:
        raise ValueError("Active NPZ set and cleaned manifest differ")
    plan.update(
        applied=True,
        backup_directory=str(backup),
        unflagged_identity_verified=True,
        installed_hashes_verified=True,
        changed_outputs_native_fk_validated=True,
        source_csvs_modified=False,
    )
    write_json(work / "plan.json", plan)
    write_json(
        dataset / "temporal_cleaning.json", {k: v for k, v in plan.items() if k != "repairs"}
    )
    print(
        f"[APPLIED] {plan['before_clips']} -> {plan['after_clips']} clips; "
        f"{plan['removed_frames']} frames removed. Backup: {backup}",
        flush=True,
    )


def verify_one(job):
    path, expected_frames, repaired = job
    with np.load(path) as f:
        arrays = {k: f[k] for k in f.files}
    from supermimic.data.motion_convert import _so3_derivative
    from supermimic.robots.profiles import CASBOT_TRACKED

    frames = len(arrays["joint_pos"])
    assert frames == expected_frames, path
    assert float(arrays["fps"].item()) == 50.0, path
    assert tuple(arrays["body_names"]) == CASBOT_TRACKED, path
    for key in TIME_KEYS:
        assert len(arrays[key]) == frames and np.isfinite(arrays[key]).all(), (path, key)
    assert arrays["joint_pos"].shape == arrays["joint_vel"].shape == (frames, 27), path
    assert arrays["body_pos_w"].shape == (frames, 14, 3), path
    assert arrays["body_lin_vel_w"].shape == arrays["body_ang_vel_w"].shape == (frames, 14, 3), path
    assert arrays["body_quat_w"].shape == (frames, 14, 4), path
    np.testing.assert_allclose(np.linalg.norm(arrays["body_quat_w"], axis=-1), 1, atol=2e-6, rtol=0)
    if len(jump_edges(arrays)):
        raise ValueError(f"Residual jump: {path}")
    root = list(arrays["body_names"]).index("base_link")
    if repaired:
        assert frames >= 51, path
        np.testing.assert_allclose(
            arrays["joint_vel"],
            np.gradient(arrays["joint_pos"].astype(np.float64), 0.02, axis=0),
            atol=1e-5,
            rtol=3e-6,
        )
        np.testing.assert_allclose(
            arrays["body_lin_vel_w"][:, root],
            np.gradient(arrays["body_pos_w"][:, root].astype(np.float64), 0.02, axis=0),
            atol=1e-5,
            rtol=3e-6,
        )
        np.testing.assert_allclose(
            arrays["body_ang_vel_w"][:, root],
            _so3_derivative(arrays["body_quat_w"][:, root].astype(np.float64), 0.02),
            atol=2e-5,
            rtol=3e-6,
        )
    return (
        frames,
        float(abs(arrays["joint_vel"]).max()),
        float(np.linalg.norm(arrays["body_lin_vel_w"][:, root], axis=-1).max()),
        float(np.linalg.norm(arrays["body_ang_vel_w"][:, root], axis=-1).max()),
    )


def verify(work, workers):
    plan = json.loads((work / "plan.json").read_text())
    if not plan["applied"]:
        raise ValueError("Apply the staged repairs before full verification")
    dataset = Path(plan["dataset"])
    repaired = {o["name"] for r in plan["repairs"] for o in r["outputs"]}
    with Path(plan["manifest"]).open() as stream:
        rows = [r for r in csv.DictReader(stream) if r["keep"] == "1"]
    jobs = [(dataset / f"{r['name']}.npz", int(r["frames"]), r["name"] in repaired) for r in rows]
    total_frames = 0
    maxima = np.zeros(3)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(verify_one, jobs, chunksize=128), 1):
            total_frames += result[0]
            maxima = np.maximum(maxima, result[1:])
            if i % 20000 == 0:
                print(f"[VERIFIED] {i}/{len(jobs)}", flush=True)
    assert total_frames == plan["after_frames"] and len(jobs) == plan["after_clips"]
    result = {
        "clips": len(jobs),
        "frames": total_frames,
        "remaining_jump_edges": 0,
        "all_time_arrays_finite": True,
        "all_quaternions_unit": True,
        "repaired_derivatives_verified": len(repaired),
        "repaired_native_fk_validated_at_staging": True,
        "max_joint_speed_rad_s": float(maxima[0]),
        "max_root_speed_m_s": float(maxima[1]),
        "max_root_angular_speed_rad_s": float(maxima[2]),
        "thresholds": THRESHOLDS,
    }
    write_json(work / "validation.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--min-seconds", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.apply and args.verify:
        parser.error("Use --apply and --verify separately")
    if args.verify:
        verify(args.work_dir.resolve(), args.workers)
    elif args.apply:
        apply(args.work_dir.resolve())
    else:
        if not all((args.dataset, args.manifest, args.audit)):
            parser.error("Staging needs --dataset, --manifest and --audit")
        if not math.isfinite(args.min_seconds) or args.min_seconds < 0.06 or args.workers < 1:
            parser.error("Need min-seconds >= 0.06 and workers >= 1")
        stage(args)


if __name__ == "__main__":
    main()
