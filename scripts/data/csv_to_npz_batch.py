"""Convert LAFAN1/BONES-SEED G1 or Casbot02 CSVs using CPU MuJoCo FK."""

from __future__ import annotations

import argparse
import csv
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from supermimic.data.motion_convert import convert_csv
from supermimic.robots import get_robot

_model = None


def _convert(job):
    global _model
    source, output, input_fps, output_fps, robot = job
    if _model is None:
        _model = get_robot(robot).model()
    try:
        frames = convert_csv(source, output, input_fps, output_fps, model=_model, robot=robot)
    except Exception as exc:
        raise RuntimeError(f"Conversion failed for {source}: {exc}") from exc
    return Path(source).name, frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("g1", "casbot02"), default="g1")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory of G1 CSVs")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--input-fps", type=float, default=30, help="LAFAN1 source rate: 30")
    parser.add_argument("--output-fps", type=float, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fps-map", type=Path, help="CSV name,fps; per-clip source FPS overrides")
    parser.add_argument(
        "--pattern", default="*.csv", help="E.g. walk1_subject1.csv for a smoke run"
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output clips")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if not all(math.isfinite(fps) and fps > 0 for fps in (args.input_fps, args.output_fps)):
        parser.error("Frame rates must be finite and positive")
    fps_by_name = {}
    if args.fps_map is not None:
        with args.fps_map.expanduser().open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"name", "fps"}.issubset(reader.fieldnames or []):
                parser.error("--fps-map requires name,fps columns")
            for row in reader:
                name, fps = row["name"], float(row["fps"])
                if name in fps_by_name or not math.isfinite(fps) or fps <= 0:
                    parser.error(f"Invalid or duplicate FPS entry: {name}")
                fps_by_name[name] = fps
    source = args.input_dir.expanduser().resolve()
    output = (
        (
            args.output_dir
            or Path("data/npz/lafan1" if args.robot == "g1" else "data/npz/lafan1_casbot02")
        )
        .expanduser()
        .resolve()
    )
    files = sorted(source.glob(args.pattern))
    if not files:
        parser.error(f"No matching CSVs in {source}")
    if len({path.stem for path in files}) != len(files):
        parser.error("Input files have duplicate output names")
    jobs = [
        (str(p), str(output / f"{p.stem}.npz"), fps_by_name.get(p.stem, args.input_fps),
         args.output_fps, args.robot)
        for p in files
        if args.overwrite or not (output / f"{p.stem}.npz").exists()
    ]
    print(
        f"Found {len(files)} CSVs; converting {len(jobs)}, skipping {len(files) - len(jobs)}.",
        flush=True,
    )
    if not jobs:
        return
    total = 0
    if args.workers == 1:
        for name, frames in map(_convert, jobs):
            total += frames
            print(f"{name}: {frames} frames", flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for name, frames in pool.map(_convert, jobs):
                total += frames
                print(f"{name}: {frames} frames", flush=True)
    print(
        f"Wrote {len(jobs)} clips, {total} frames "
        f"({total / args.output_fps / 3600:.3f} hours) to {output}"
    )


if __name__ == "__main__":
    main()
