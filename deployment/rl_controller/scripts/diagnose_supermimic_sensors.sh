#!/usr/bin/env bash
# Passive SDK receive test. Does not start a controller or publish motor commands.
set -eo pipefail
SM_DIAG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/humble/setup.bash
export LD_LIBRARY_PATH="$SM_DIAG_ROOT/lib:${LD_LIBRARY_PATH:-}"
SM_DIAG_PROBE="$SM_DIAG_ROOT/bin/supermimic_sensor_probe"
if [[ ! -x "$SM_DIAG_PROBE" ]]; then
    SM_DIAG_PROBE="$SM_DIAG_ROOT/build/supermimic_ros/supermimic_sensor_probe"
fi
if [[ ! -x "$SM_DIAG_PROBE" ]]; then
    echo "Missing sensor probe; rebuild the deployment package." >&2
    exit 1
fi
if [[ "${1:-}" == "--udp" ]]; then
    shift
    SM_DIAG_PROFILE="$SM_DIAG_ROOT/config/supermimic_udp_only.xml"
    if [[ ! -r "$SM_DIAG_PROFILE" ]]; then
        echo "Missing UDP profile: $SM_DIAG_PROFILE" >&2
        exit 1
    fi
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    export FASTRTPS_DEFAULT_PROFILES_FILE="$SM_DIAG_PROFILE"
    export FASTDDS_DEFAULT_PROFILES_FILE="$SM_DIAG_PROFILE"
    echo "[INFO] Probe transport: UDPv4 only, profile=$SM_DIAG_PROFILE"
fi
exec "$SM_DIAG_PROBE" "$@"
