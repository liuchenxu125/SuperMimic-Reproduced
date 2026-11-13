#!/usr/bin/env bash
# Local native MuJoCo only; does not start ROS or send hardware commands.
# cpu9 / 92000 actor: 930 history + 690 goal; updated 3mm SEED references, same bank as hardware.
set -euo pipefail
SM_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SM_PROJECT_ROOT"
exec env -u PYTHONPATH MUJOCO_GL=glfw uv run python scripts/sim2sim.py \
  --bundle outputs/casbot02_deployment/cpu9_092000_updated_seed_20261112/sim2sim \
  --clip 踏步 \
  --motion-order-config rl_controller/config/supermimic_walk1_subject2_proprio.yaml \
  --render "$@"
