#!/usr/bin/env python3
"""Independent frozen-teacher human encoder training; never invokes sm-train."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-ckpt", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=60000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--masking", choices=["mixed", "full", "five"], default="mixed")
    parser.add_argument("--obs-noise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--loss-normalization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--action-weight", type=float, default=1.0)
    parser.add_argument("--align-weight", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=4000)
    parser.add_argument(
        "--cache-clips",
        type=int,
        default=4,
        help="CPU clip cache; 0 keeps all selected pairs in RAM",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    # Set these before NumPy/Torch import, only in this new process.
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = str(args.cpu_threads)
    from supermimic.human.training import HumanTrainConfig, HumanTrainer

    HumanTrainer(HumanTrainConfig(**vars(args))).run()


if __name__ == "__main__":
    main()
