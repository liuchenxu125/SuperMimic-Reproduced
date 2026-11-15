#!/usr/bin/env bash
set -euo pipefail
SM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ ! -x "$SM_ROOT/.venv-pico/bin/python" ]]; then
    echo "Run bash $SM_ROOT/scripts/teleop/setup_pico_env.sh first" >&2
    exit 1
fi
cd "$SM_ROOT"
exec env -u PYTHONPATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$SM_ROOT/.venv-pico/bin/python" "$SM_ROOT/scripts/teleop/stream_pico.py" "$@"
