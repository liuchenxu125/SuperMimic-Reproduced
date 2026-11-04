#!/usr/bin/env bash
# Local native MuJoCo only; does not start ROS or send hardware commands.
set -euo pipefail
SM_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SM_PROJECT_ROOT"
exec env -u PYTHONPATH MUJOCO_GL=glfw uv run python scripts/sim2sim.py \
  --bundle outputs/casbot02_bones_seed_cpu4_092000_deployment_20261104/sim2sim \
  --clip 踏步 \
  --motion-order 踏步 TongYongDongTaiQingTian_casbot02-bass1_6R7R2R1R_hand01_whole_50hz_v2 BS_XiaoDongTai_TaiJiaoPanDing_100HZ X1_1000HZ SEED_鼓掌 SEED_看时间 SEED_思考 SEED_捂脸 SEED_扭胯 SEED_武术 SEED_摆胯 SEED_迪斯科 dance_phew_001__A464 dance_click_002__A464 dance_floss_002__A464 dance_fresh_001__A464 dance_fishin_002__A464 dance_goated_001__A464 dance_macarena_003__A464 dance_peace_out_002__A464 dance_peace_out_004__A464 dance_pirouette_001__A464 dance_wavy_worm_001__A467 后退 原地左转 原地右转 \
  --render "$@"
