#!/usr/bin/env bash
# Separate PICO/IK environment: never uv sync or pip install into training .venv.
set -euo pipefail
SM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SM_ENV="$SM_ROOT/.venv-pico"
SM_XRT_REF=8941e1dbbf4313fe83846f82984eaa8c992690bf
SM_XRT="$SM_ROOT/outputs/pico_sdk/$SM_XRT_REF"
command -v uv >/dev/null
command -v g++ >/dev/null
if [[ ! -x "$SM_ENV/bin/python" ]]; then
    uv venv --python "$SM_ROOT/.venv/bin/python" "$SM_ENV"
fi
SM_MUJOCO_VERSION=$("$SM_ROOT/.venv/bin/python" -c 'import importlib.metadata as m; print(m.version("mujoco"))')
uv pip install --python "$SM_ENV/bin/python" numpy scipy PyYAML "mujoco==$SM_MUJOCO_VERSION"
uv pip install --python "$SM_ENV/bin/python" --no-deps -e "$SM_ROOT"
mkdir -p "$(dirname "$SM_XRT")"
if [[ ! -d "$SM_XRT/.git" ]]; then
    git clone --filter=blob:none https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind "$SM_XRT"
fi
if [[ $(git -C "$SM_XRT" rev-parse HEAD) != "$SM_XRT_REF" ]]; then
    git -C "$SM_XRT" fetch origin "$SM_XRT_REF"
    git -C "$SM_XRT" checkout --detach "$SM_XRT_REF"
fi
uv pip install --python "$SM_ENV/bin/python" "$SM_XRT"
"$SM_ENV/bin/python" -c 'import xrobotoolkit_sdk; from supermimic.teleop.pico import Resampler; print("PICO SDK import OK; hardware not initialized")'
echo "Independent environment ready: $SM_ENV"
