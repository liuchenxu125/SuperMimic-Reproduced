"""CPU reference-library environment bootstrap; keep this module torch-free."""
from __future__ import annotations

import argparse
import os
from collections.abc import MutableMapping, Sequence


CPU4_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def configure_cpu_reference_environment(
    argv: Sequence[str], environ: MutableMapping[str, str] | None = None
) -> bool:
    """Apply cpu4's limits before importing Torch/NumPy/BLAS on the CLI path.

    Parse only the reference storage switch; tyro remains responsible for the
    full command. GPU-resident reference jobs keep their existing environment.
    """
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--env.library-device", dest="library_device")
    args, _ = parser.parse_known_args(list(argv))
    if args.library_device != "cpu":
        return False
    target = os.environ if environ is None else environ
    for key in CPU4_THREAD_ENV:
        target[key] = "1"
    return True
