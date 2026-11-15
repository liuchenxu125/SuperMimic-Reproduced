#!/usr/bin/env bash
# Local official platform-tools; leave system packages and the training env alone.
set -euo pipefail
SM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "$SM_ROOT/outputs/pico_setup/platform-tools/adb" "$@"
