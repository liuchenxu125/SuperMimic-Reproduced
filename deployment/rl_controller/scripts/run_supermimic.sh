#!/usr/bin/env bash
# Does not switch SDK mode or kill processes. With auto_start enabled, an
# queried ZERO/STAND followed by fresh raw topic mode 7 starts the preview.
# SDK 2.2.37: actual gains must be enabled (use_default_kp_kd=false) and echoed.
set -eo pipefail
SM_RUN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/humble/setup.bash
if [[ -f "$SM_RUN_ROOT/setup.bash" ]]; then
    source "$SM_RUN_ROOT/setup.bash"
fi
if [[ -f "$SM_RUN_ROOT/build/supermimic_msgs/share/crb_ros_msg/local_setup.bash" ]]; then
    source "$SM_RUN_ROOT/build/supermimic_msgs/share/crb_ros_msg/local_setup.bash"
fi
export LD_LIBRARY_PATH="$SM_RUN_ROOT/lib:${LD_LIBRARY_PATH:-}"
SM_RUNNER="$SM_RUN_ROOT/bin/supermimic_runner"
if [[ ! -x "$SM_RUNNER" ]]; then
    SM_RUNNER="$SM_RUN_ROOT/build/supermimic_ros/supermimic_runner"
fi
SM_CONFIG="${SUPERMIMIC_CONFIG:-$SM_RUN_ROOT/config/supermimic_walk1_subject2_proprio.yaml}"
if [[ ! -x "$SM_RUNNER" ]]; then
    echo "SuperMimic runner missing; run scripts/build_supermimic.sh first." >&2
    exit 1
fi
exec "$SM_RUNNER" "$SM_CONFIG" "$@"
