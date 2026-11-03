"""Filter BONES-SEED NPZs or prepare Casbot02 CSVs as accepted training NPZs.

Keywords/thresholds follow CuboiLeo/SuperMimic-Reproduced ba61053, including
acrobatics exclusions. Casbot02 uses its own primitive foot geometry and a
stature-adjusted static-upright height gate. NPZ mode never modifies motions;
CSV mode writes new accepted NPZs. Pass --data.manifest explicitly in training.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing
import os
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

from supermimic.data.bones_seed import filter_clip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--npz-dir", type=Path)
    inputs.add_argument("--csv-dir", type=Path, help="Casbot02: filter names/FK feasibility and convert accepted CSVs")
    parser.add_argument("--robot", choices=("g1", "casbot02"), default="g1")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--input-fps", type=float)
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--limit", type=int, help="Evenly spaced pilot subset; use a separate output directory")
    parser.add_argument("--tracked-bodies-only", action="store_true", help="Store the 14 training bodies in float32")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")
    if args.out is not None and args.out.suffix.lower() != ".csv":
        parser.error("--out must be a CSV manifest; motion NPZs must not be overwritten")
    if args.csv_dir is not None:
        from supermimic.data.bones_preparation import prepare_csv_dataset

        prepare_csv_dataset(args)
        return
    files = sorted(args.npz_dir.expanduser().glob("*.npz"))
    if not files:
        parser.error("No NPZ clips found")
    output = (args.out or args.npz_dir.expanduser() / "manifest_filtered.csv").expanduser()
    if output.suffix.lower() != ".csv":
        parser.error("--out must be a CSV manifest; motion NPZs must not be overwritten")

    def collect(results):
        rows = []
        for row in results:
            rows.append(row)
            if len(rows) % args.progress_every == 0 or len(rows) == len(files):
                print(f"Filtered {len(rows)}/{len(files)}", flush=True)
        return rows

    print(f"Checking {len(files)} {args.robot} NPZ clips", flush=True)
    worker = partial(filter_clip, robot=args.robot)
    if args.workers == 1:
        rows = collect(map(worker, files))
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            rows = collect(pool.map(worker, files, chunksize=32))
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=output.parent, suffix=".csv")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["name", "frames", "keep", "reason"])
            writer.writerows(
                (name, frames, int(reason is None), reason or "") for name, frames, reason in rows
            )
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    kept = sum(reason is None for _, _, reason in rows)
    hours = sum(frames for _, frames, reason in rows if reason is None) / 50 / 3600
    print(f"{args.robot}: {len(rows)} clips -> {kept} kept ({hours:.3f} h)")
    print("Excluded:", dict(Counter(reason for _, _, reason in rows if reason)))
    print(f"Saved {output}; pass this manifest explicitly to training/evaluation")


if __name__ == "__main__":
    main()
