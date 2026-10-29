"""Write a G1 BONES-SEED inclusion manifest; NPZs are never modified.

Keywords/thresholds follow CuboiLeo/SuperMimic-Reproduced ba61053, including
acrobatics exclusions. Use --data.manifest in training to opt into the result.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing
import os
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from supermimic.data.bones_seed import filter_clip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    files = sorted(args.npz_dir.expanduser().glob("*.npz"))
    if not files:
        parser.error("No NPZ clips found")
    output = (args.out or args.npz_dir.expanduser() / "manifest_filtered.csv").expanduser()
    if output.suffix.lower() != ".csv":
        parser.error("--out must be a CSV manifest; motion NPZs must not be overwritten")
    if args.workers == 1:
        rows = list(map(filter_clip, files))
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            rows = list(pool.map(filter_clip, files, chunksize=32))
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
    print(f"G1: {len(rows)} clips -> {kept} kept ({hours:.3f} h)")
    print("Excluded:", dict(Counter(reason for _, _, reason in rows if reason)))
    print(f"Saved {output}; pass this manifest explicitly to training/evaluation")


if __name__ == "__main__":
    main()
