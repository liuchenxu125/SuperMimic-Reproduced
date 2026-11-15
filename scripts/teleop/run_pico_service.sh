#!/usr/bin/env bash
# Run the official, locally extracted PC Service without system-wide installation.
set -euo pipefail
SM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SM_PICO_SERVICE="$SM_ROOT/outputs/pico_setup/pc-service/opt/apps/roboticsservice"
if [[ ! -x "$SM_PICO_SERVICE/RoboticsServiceProcess" ]]; then
    echo "PC Service missing; see docs/PICO_TELEOP.md (first connection)." >&2
    exit 1
fi
export LD_LIBRARY_PATH="$SM_PICO_SERVICE:$SM_PICO_SERVICE/lib:$SM_PICO_SERVICE/SDK/x64:${LD_LIBRARY_PATH:-}"
export QT_PLUGIN_PATH="$SM_PICO_SERVICE/plugins:${QT_PLUGIN_PATH:-}"
export QT_QML_PATH="$SM_PICO_SERVICE/qml:${QT_QML_PATH:-}"
cd "$SM_PICO_SERVICE"
exec ./RoboticsServiceProcess
