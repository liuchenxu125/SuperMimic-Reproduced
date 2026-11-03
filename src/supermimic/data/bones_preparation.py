"""Resumable Casbot02 BONES CSV -> feasibility filter -> training NPZ pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
import multiprocessing
import os
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .bones_seed import casbot_foot_model, feasibility_arrays, name_filter
from .motion_convert import csv_motion_arrays, write_motion_npz

_MODEL = None
_OPTIONS = None


def _initialize(options):
    global _MODEL, _OPTIONS
    from supermimic.robots import get_robot

    _OPTIONS = options
    _MODEL = get_robot("casbot02").model(collision_model="primitive")


def _prepare(job):
    source, size, mtime = job
    source = Path(source)
    row = {
        "name": source.stem,
        "source": str(source),
        "source_bytes": size,
        "source_mtime_ns": mtime,
        "frames": 0,
        "keep": 0,
        "reason": "",
        "npz_bytes": 0,
    }
    output = Path(_OPTIONS["output"]) / f"{source.stem}.npz"
    try:
        arrays = csv_motion_arrays(source, _OPTIONS["input_fps"], 50, _MODEL, "casbot02")
        row["frames"] = len(arrays["joint_pos"])
        row["reason"] = feasibility_arrays(arrays, "casbot02") or ""
        if row["reason"]:
            return row
        if _OPTIONS["tracked_only"]:
            from supermimic.robots import get_robot

            names = list(arrays["body_names"])
            indices = [names.index(n) for n in get_robot("casbot02").tracked_bodies]
            for key in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
                arrays[key] = arrays[key][:, indices]
            arrays["body_names"] = arrays["body_names"][indices]
        if shutil.disk_usage(output.parent).free < 2 * 1024**3:
            raise OSError("Less than 2 GiB free; stopping resumably")
        write_motion_npz(output, arrays)
        row.update(keep=1, npz_bytes=output.stat().st_size)
    except OSError:
        raise
    except (ValueError, TypeError, RuntimeError, FloatingPointError, KeyError) as exc:
        row["reason"] = f"invalid:{type(exc).__name__}:{exc}"
    return row


def _write_manifest(path, rows):
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        fields = [
            "name",
            "frames",
            "keep",
            "reason",
            "source",
            "source_bytes",
            "source_mtime_ns",
            "npz_bytes",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["name"]))
    os.replace(temporary, path)


def prepare_csv_dataset(args):
    """Filter all CSV names, compute full FK, and store accepted motions only."""
    from supermimic.robots.profiles import CASBOT_ASSETS

    if args.robot != "casbot02":
        raise ValueError(
            "CSV preparation is currently Casbot02-only; G1 uses the existing NPZ workflow"
        )
    if (
        args.output_dir is None
        or args.input_fps is None
        or not np.isfinite(args.input_fps)
        or args.input_fps <= 0
    ):
        raise ValueError("CSV mode requires --output-dir and positive --input-fps")
    source = args.csv_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    files = sorted(source.glob(args.pattern))
    if args.limit:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        files = [
            files[i] for i in np.linspace(0, len(files) - 1, min(args.limit, len(files)), dtype=int)
        ]
    if not files or len({p.stem for p in files}) != len(files):
        raise ValueError("Need CSV files with unique output names")
    inventory = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files]
    inventory_hash = hashlib.sha256(json.dumps(inventory).encode()).hexdigest()
    options = {
        "output": str(output),
        "source": str(source),
        "input_fps": args.input_fps,
        "output_fps": 50,
        "tracked_only": args.tracked_bodies_only,
        "source_inventory_sha256": inventory_hash,
        "primitive_xml_sha256": hashlib.sha256(
            (CASBOT_ASSETS / "xml/casbot02_primitives.xml").read_bytes()
        ).hexdigest(),
        "motors_sha256": hashlib.sha256(
            (CASBOT_ASSETS / "motor_parameters.json").read_bytes()
        ).hexdigest(),
    }
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "preparation.json"
    journal_path = output / "preparation.jsonl"
    if config_path.exists():
        if json.loads(config_path.read_text()) != options:
            raise ValueError(
                "Preparation inputs/options changed; choose a different output directory"
            )
    else:
        if any(output.glob("*.npz")):
            raise ValueError("Refusing to mix untracked existing NPZs into a new preparation")
        config_path.write_text(json.dumps(options, indent=2))
    completed = {}
    if journal_path.exists():
        # Only newline-terminated records were committed. A killed final append
        # is discarded before resuming, so it cannot corrupt the next record.
        with journal_path.open("rb+") as stream:
            while True:
                begin = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    stream.truncate(begin)
                    break
                row = json.loads(line)
                path = output / f"{row['name']}.npz"
                if (not row["keep"] and not row["reason"].startswith("invalid:")) or (
                    row["keep"] and path.exists() and path.stat().st_size == row["npz_bytes"]
                ):
                    completed[row["name"]] = row
    jobs, rows = [], {}
    for src, size, mtime in inventory:
        p = Path(src)
        if p.stem in completed:
            rows[p.stem] = completed[p.stem]
            continue
        reason = name_filter(p.stem)
        if reason:
            rows[p.stem] = {
                "name": p.stem,
                "frames": 0,
                "keep": 0,
                "reason": reason,
                "source": src,
                "source_bytes": size,
                "source_mtime_ns": mtime,
                "npz_bytes": 0,
            }
        else:
            jobs.append((src, size, mtime))
    # Retargeted CSVs were atomically written as they completed. Their creation
    # chronology is a useful locality hint on external disks; lexical names
    # interleave recordings and force unnecessary random reads.
    jobs.sort(key=lambda job: job[2])
    start = time.monotonic()
    nominal_height = casbot_foot_model()[2]
    print(
        f"Casbot02: {len(files)} CSVs; {len(jobs)} pending conversions; {len(completed)} resumed",
        flush=True,
    )
    print(
        f"120/other source FPS={args.input_fps}; output=50 Hz; nominal standing height={nominal_height:.6f} m",
        flush=True,
    )
    print(
        f"Static-upright height gate={nominal_height * 0.60 / 0.78:.6f}..{nominal_height * 0.85 / 0.78:.6f} m; foot clearance=0.02 m",
        flush=True,
    )
    with journal_path.open("a", encoding="utf-8", buffering=1) as journal:
        for row in rows.values():
            if row["name"] not in completed:
                journal.write(json.dumps(row, ensure_ascii=False) + "\n")
        options_worker = dict(options)
        context = multiprocessing.get_context("spawn")
        with context.Pool(
            args.workers, initializer=_initialize, initargs=(options_worker,)
        ) as pool:
            for index, row in enumerate(pool.imap_unordered(_prepare, jobs, chunksize=8), 1):
                rows[row["name"]] = row
                journal.write(json.dumps(row, ensure_ascii=False) + "\n")
                if index % args.progress_every == 0 or index == len(jobs):
                    rate = index / max(time.monotonic() - start, 0.001)
                    eta = (len(jobs) - index) / rate
                    print(
                        f"Prepared {index}/{len(jobs)}; {rate:.1f} clips/s; ETA={eta / 60:.1f} min; "
                        f"last={row['name']} keep={row['keep']} reason={row['reason']}",
                        flush=True,
                    )
    values = list(rows.values())
    manifest = (args.out or output / "manifest_filtered.csv").expanduser()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    _write_manifest(manifest, values)
    kept = [row for row in values if row["keep"]]
    reasons = dict(Counter(row["reason"] for row in values if not row["keep"]))
    summary = dict(
        status="complete"
        if not any(r.startswith("invalid:") for r in reasons)
        else "complete_with_invalid_sources",
        robot="casbot02",
        source_csv_clips=len(files),
        kept_clips=len(kept),
        kept_frames=sum(r["frames"] for r in kept),
        kept_hours=sum(r["frames"] for r in kept) / 50 / 3600,
        kept_time_span_hours=sum(r["frames"] - 1 for r in kept) / 50 / 3600,
        npz_bytes=sum(r["npz_bytes"] for r in kept),
        reasons=reasons,
        manifest=str(manifest),
        **options,
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
