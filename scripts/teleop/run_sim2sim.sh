#!/usr/bin/env bash
set -euo pipefail
SM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SM_ROOT"
exec env -u PYTHONPATH MUJOCO_GL=glfw OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$SM_ROOT/.venv/bin/python" "$SM_ROOT/scripts/teleop/sim2sim_live.py" --render "$@"
