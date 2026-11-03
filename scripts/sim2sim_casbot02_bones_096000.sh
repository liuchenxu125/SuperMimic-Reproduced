#!/usr/bin/env bash
# Local native MuJoCo only; does not start ROS or send hardware commands.
set -euo pipefail
SM_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SM_PROJECT_ROOT"
exec env -u PYTHONPATH MUJOCO_GL=glfw uv run python scripts/sim2sim.py \
  --bundle outputs/casbot02_bones_seed_096000_deployment_20261102/sim2sim \
  --clip 踏步 \
  --motion-order 踏步 BS_XiaoDongTai_TaiJiaoPanDing_100HZ X1_1000HZ SEED_鼓掌 SEED_看时间 SEED_思考 SEED_捂脸 SEED_扭胯 SEED_武术 SEED_摆胯 SEED_迪斯科 后退 原地左转 原地右转 \
  --render "$@"
