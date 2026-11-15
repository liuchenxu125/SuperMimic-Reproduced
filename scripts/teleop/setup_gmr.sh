#!/usr/bin/env bash
# Optional full-body GMR; keep all dependencies out of the training .venv.
set -euo pipefail
SM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SM_ENV="$SM_ROOT/.venv-pico"
SM_GMR_REF=703841aa354920715f4e6f84d7080016699af1ac
SM_GMR_ROOT="${1:-$HOME/Desktop/GMR}"
if [[ ! -x "$SM_ENV/bin/python" ]]; then
    echo "First run bash scripts/teleop/setup_pico_env.sh" >&2
    exit 1
fi
if [[ ! -f "$SM_GMR_ROOT/general_motion_retargeting/motion_retarget.py" ]]; then
    # New checkouts live inside the project by default; do not change an existing GMR checkout.
    SM_GMR_ROOT="$SM_ROOT/outputs/teleop_dependencies/GMR-$SM_GMR_REF"
    mkdir -p "$(dirname "$SM_GMR_ROOT")"
    git clone --filter=blob:none https://github.com/YanjieZe/GMR.git "$SM_GMR_ROOT"
    git -C "$SM_GMR_ROOT" checkout --detach "$SM_GMR_REF"
fi
SM_MUJOCO_VERSION=$("$SM_ROOT/.venv/bin/python" -c 'import importlib.metadata as m; print(m.version("mujoco"))')
uv pip install --python "$SM_ENV/bin/python" "mujoco==$SM_MUJOCO_VERSION" \
    mink==1.3.0 qpsolvers==4.13.0 daqp==0.10.3 rich==14.3.4
"$SM_ENV/bin/python" -c 'import mink, daqp; print("GMR dependencies ready; no headset or motor connection")'
echo "GMR source: $SM_GMR_ROOT"
echo "Sender: bash scripts/teleop/run_pico.sh --retargeter gmr --input-mode full-body --gmr-root $SM_GMR_ROOT"
