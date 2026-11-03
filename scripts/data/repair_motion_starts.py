"""Stage/apply an additive Casbot02 reference repair variant from reviewed evidence.

The parent dataset and its manifests stay byte-identical. The new directory uses
hard links for unaffected NPZs; all repaired NPZs are independent atomic writes.
Future data tools must replace files atomically rather than edit hard links in place.
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

from supermimic.data.geometry_cleaning import (
    collision_depths,
    collision_model_with_floor,
    compatible_spans,
)
from supermimic.data.motion_convert import quat_rotate_inverse, write_motion_npz
from supermimic.data.temporal_cleaning import rebuild_span

MODEL = None
OPTIONS = None


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, data):
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def manifest_rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def write_manifest(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    temp = path.with_suffix(".csv.tmp")
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["name"]))
    os.replace(temp, path)


def initialize(options):
    global MODEL, OPTIONS
    MODEL, OPTIONS = collision_model_with_floor(), options


def native_validate(arrays):
    data = mujoco.MjData(MODEL)
    root = list(arrays["body_names"]).index("base_link")
    ids = [MODEL.body(str(n)).id for n in arrays["body_names"]]
    for t in np.unique(np.linspace(0, len(arrays["joint_pos"]) - 1, 5, dtype=int)):
        data.qpos[:] = np.r_[
            arrays["body_pos_w"][t, root], arrays["body_quat_w"][t, root], arrays["joint_pos"][t]
        ]
        data.qvel[:] = np.r_[
            arrays["body_lin_vel_w"][t, root],
            quat_rotate_inverse(data.qpos[3:7], arrays["body_ang_vel_w"][t, root]),
            arrays["joint_vel"][t],
        ]
        mujoco.mj_forward(MODEL, data)
        np.testing.assert_allclose(arrays["body_pos_w"][t], data.xpos[ids], atol=3e-6, rtol=2e-6)
        for b, body_id in enumerate(ids):
            velocity = np.empty(6)
            mujoco.mj_objectVelocity(MODEL, data, mujoco.mjtObj.mjOBJ_XBODY, body_id, velocity, 0)
            np.testing.assert_allclose(
                arrays["body_ang_vel_w"][t, b], velocity[:3], atol=2e-5, rtol=3e-6
            )
            np.testing.assert_allclose(
                arrays["body_lin_vel_w"][t, b], velocity[3:], atol=2e-5, rtol=3e-6
            )


def repair(job):
    row, reasons = job
    name = row["name"]
    parent = Path(OPTIONS["dataset"]) / f"{name}.npz"
    original_sha = digest(parent)
    with np.load(parent) as f:
        arrays = {k: f[k] for k in f.files}
    ground, own = collision_depths(arrays, MODEL)
    proof = Path(OPTIONS["audit_dir"]) / "frame_geometry" / parent.name
    with np.load(proof) as f:
        np.testing.assert_allclose(ground, f["ground_depth"], atol=1e-7, rtol=0)
        np.testing.assert_allclose(own, f["self_depth"], atol=1e-7, rtol=0)
    prefix = 5 if any(r.startswith("verified_") for r in reasons) else 0
    spans = compatible_spans(
        ground,
        own,
        trim_prefix=prefix,
        max_depth=OPTIONS["max_depth"],
        min_frames=OPTIONS["min_frames"],
    )
    source = Path(str(arrays["source"].item()))
    if source.stat().st_size != int(row["source_bytes"]) or source.stat().st_mtime_ns != int(
        row["source_mtime_ns"]
    ):
        raise ValueError(f"Original CSV changed: {source}")
    result = {
        "name": name,
        "original_sha256": original_sha,
        "frames": len(ground),
        "reasons": reasons,
        "prefix_trim_frames": prefix,
        "invalid_geometry_frames": int(
            ((ground > OPTIONS["max_depth"]) | (own > OPTIONS["max_depth"])).sum()
        ),
        "source_csv": str(source),
        "source_csv_sha256": digest(source),
        "outputs": [],
    }
    original_start = int(arrays.get("temporal_source_frame_start", np.array(0)).item())
    for i, (start, end) in enumerate(spans, 1):
        new_name = f"{name}__startfix{i:02d}"
        cleaned = rebuild_span(arrays, MODEL, start, end)
        cleaned.update(
            startfix_version=np.array(1),
            startfix_parent_name=np.array(name),
            startfix_parent_sha256=np.array(original_sha),
            startfix_reasons=np.array(",".join(reasons)),
            startfix_parent_frame_start=np.array(start),
            startfix_parent_frame_end=np.array(end),
            temporal_source_frame_start=np.array(original_start + start),
            temporal_source_frame_end=np.array(original_start + end),
        )
        new_ground, new_own = collision_depths(cleaned, MODEL)
        if max(new_ground.max(), new_own.max()) > OPTIONS["max_depth"] + 2e-6:
            raise ValueError(f"Rebuilt output has incompatible geometry: {new_name}")
        native_validate(cleaned)
        output = Path(OPTIONS["work"]) / "staged" / f"{new_name}.npz"
        write_motion_npz(output, cleaned)
        with np.load(output) as f:
            for key in cleaned:
                np.testing.assert_array_equal(f[key], cleaned[key])
        result["outputs"].append(
            {
                "name": new_name,
                "start": start,
                "end": end,
                "frames": end - start,
                "original_start": original_start + start,
                "original_end": original_start + end,
                "sha256": digest(output),
                "bytes": output.stat().st_size,
                "max_ground_depth_m": float(new_ground.max()),
                "max_self_depth_m": float(new_own.max()),
            }
        )
    return result


def stage(args):
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    if (work / "plan.json").exists():
        raise FileExistsError("Plan exists; inspect and use --apply")
    if args.output.resolve() == args.dataset.resolve():
        raise ValueError("The repaired variant must have its own directory")
    rows = manifest_rows(args.manifest)
    selected = {r["name"]: r for r in rows if r["keep"] == "1"}
    audit = args.audit_dir.resolve()
    full = json.loads((audit / "full_geometry.json").read_text())
    raw = {
        r["name"] for r in json.loads((audit / "verified_source_startup_outliers.json").read_text())
    }
    boundary = {
        r["name"] for r in json.loads((audit / "verified_boundary_outliers.json").read_text())
    }
    reasons = {}
    for row in full:
        why = []
        if row["bad_before_original_failure"]:
            why.append("primitive_geometry_incompatible_before_failure")
        if row["name"] in raw:
            why.append("verified_source_frame0_outlier")
        if row["name"] in boundary:
            why.append("verified_cut_boundary_outlier")
        if why:
            reasons[row["name"]] = why
    if not reasons or not set(reasons) <= selected.keys():
        raise ValueError("Evidence names must be selected parent motions")
    minimum = math.ceil(args.min_seconds * 50) + 1
    options = {
        "dataset": str(args.dataset.resolve()),
        "audit_dir": str(audit),
        "work": str(work),
        "max_depth": args.max_depth,
        "min_frames": minimum,
    }
    (work / "staged").mkdir(exist_ok=True)
    jobs = [(selected[name], why) for name, why in sorted(reasons.items())]
    repairs = []
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=initialize, initargs=(options,)
    ) as pool:
        for i, result in enumerate(pool.map(repair, jobs, chunksize=4), 1):
            repairs.append(result)
            if i % 25 == 0:
                print("[STAGED]", i, "/", len(jobs), flush=True)
    replacements = {r["name"]: r for r in repairs}
    updated = []
    for row in rows:
        if row["name"] not in replacements:
            updated.append(row)
            continue
        r = replacements[row["name"]]
        updated.append(
            dict(
                row,
                keep=0,
                reason="startfix_superseded" if r["outputs"] else "startfix_no_compatible_1s_span",
            )
        )
        for o in r["outputs"]:
            updated.append(
                dict(
                    row,
                    name=o["name"],
                    frames=o["frames"],
                    keep=1,
                    reason="",
                    npz_bytes=o["bytes"],
                    original_name=row.get("original_name") or row["name"],
                    parent_motion=row["name"],
                    source_frame_start=o["original_start"],
                    source_frame_end=o["original_end"],
                    temporal_cleaning=(row.get("temporal_cleaning", "") + ";startfix_v1").strip(
                        ";"
                    ),
                )
            )
    accepted = [r for r in updated if str(r["keep"]) == "1"]
    new_names = [r["name"] for r in accepted]
    if len(set(new_names)) != len(new_names):
        raise ValueError("Output names collide")
    write_manifest(work / "manifest_startfix.csv", updated)
    (work / "parent_manifest.csv").write_bytes(args.manifest.read_bytes())
    plan = {
        "version": 1,
        "applied": False,
        "dataset": str(args.dataset.resolve()),
        "output": str(args.output.resolve()),
        "parent_manifest": str(args.manifest.resolve()),
        "parent_manifest_sha256": digest(args.manifest),
        "cleaned_manifest_sha256": digest(work / "manifest_startfix.csv"),
        "max_geometry_penetration_m": args.max_depth,
        "min_frames": minimum,
        "note": "Geometry compatibility uses the current primitive model, not original retarget meshes; not a paper filter.",
        "before_clips": len(selected),
        "after_clips": len(accepted),
        "targets": len(repairs),
        "unchanged_clips": len(selected) - len(repairs),
        "repaired_parent_clips": sum(bool(r["outputs"]) for r in repairs),
        "repaired_segments": sum(len(r["outputs"]) for r in repairs),
        "quarantined_parent_clips": sum(not r["outputs"] for r in repairs),
        "before_frames": sum(int(r["frames"]) for r in selected.values()),
        "after_frames": sum(int(r["frames"]) for r in accepted),
        "repairs": repairs,
    }
    write_json(work / "plan.json", plan)
    print(json.dumps({k: v for k, v in plan.items() if k != "repairs"}, indent=2), flush=True)


def apply(work):
    plan = json.loads((work / "plan.json").read_text())
    if plan["applied"]:
        raise FileExistsError("Already applied; refusing duplicate repair")
    parent, output = Path(plan["dataset"]), Path(plan["output"])
    if output.exists():
        raise FileExistsError(
            "Variant output already exists; no unrelated files will be overwritten"
        )
    manifest = work / "manifest_startfix.csv"
    if digest(Path(plan["parent_manifest"])) != plan["parent_manifest_sha256"]:
        raise ValueError("Parent manifest changed")
    if digest(manifest) != plan["cleaned_manifest_sha256"]:
        raise ValueError("Staged manifest changed")
    for r in plan["repairs"]:
        if digest(parent / f"{r['name']}.npz") != r["original_sha256"]:
            raise ValueError(f"Parent changed: {r['name']}")
        if digest(Path(r["source_csv"])) != r["source_csv_sha256"]:
            raise ValueError(f"Original source changed: {r['name']}")
        for o in r["outputs"]:
            if digest(work / "staged" / f"{o['name']}.npz") != o["sha256"]:
                raise ValueError("Staged NPZ changed")
    output.mkdir(parents=True)
    backup = work / "backup"
    backup.mkdir(exist_ok=True)
    for r in plan["repairs"]:
        os.link(parent / f"{r['name']}.npz", backup / f"{r['name']}.npz")
    rows = manifest_rows(manifest)
    new = {o["name"]: o for r in plan["repairs"] for o in r["outputs"]}
    for row in rows:
        if row["keep"] != "1":
            continue
        target = output / f"{row['name']}.npz"
        if row["name"] in new:
            os.replace(work / "staged" / target.name, target)
            if digest(target) != new[row["name"]]["sha256"]:
                raise ValueError("Installed output differs")
        else:
            original = parent / target.name
            os.link(original, target)
            assert target.stat().st_ino == original.stat().st_ino
    for name in ("manifest_filtered.csv", "manifest_startfix.csv"):
        shutil.copy2(manifest, output / name)
    expected = {r["name"] for r in rows if r["keep"] == "1"}
    if {p.stem for p in output.glob("*.npz")} != expected:
        raise ValueError("Variant files differ from manifest")
    for r in plan["repairs"]:
        if digest(parent / f"{r['name']}.npz") != r["original_sha256"]:
            raise ValueError("Original was unexpectedly changed")
    assert digest(Path(plan["parent_manifest"])) == plan["parent_manifest_sha256"]
    plan.update(
        applied=True,
        parent_data_unchanged=True,
        source_csvs_unchanged=True,
        unchanged_clips_hardlinked=True,
    )
    write_json(work / "plan.json", plan)
    write_json(output / "summary.json", {k: v for k, v in plan.items() if k != "repairs"})
    with (work / "repaired_clips.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "parent_motion", "start", "end", "frames"])
        writer.writeheader()
        for r in plan["repairs"]:
            for o in r["outputs"]:
                writer.writerow(
                    {
                        "name": o["name"],
                        "parent_motion": r["name"],
                        "start": o["start"],
                        "end": o["end"],
                        "frames": o["frames"],
                    }
                )
    print("[APPLIED]", plan["after_clips"], "clips in", output, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--audit-dir", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-depth", type=float, default=0.03)
    p.add_argument("--min-seconds", type=float, default=1.0)
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    if args.apply:
        apply(args.work_dir.resolve())
    else:
        if not all((args.dataset, args.manifest, args.audit_dir, args.output)):
            p.error("Staging requires dataset, manifest, audit-dir and output")
        if args.workers < 1 or not math.isfinite(args.max_depth) or args.max_depth <= 0:
            p.error("Need workers >= 1 and positive finite max-depth")
        if not math.isfinite(args.min_seconds) or args.min_seconds < 0.06:
            p.error("Need min-seconds >= .06")
        stage(args)


if __name__ == "__main__":
    main()
