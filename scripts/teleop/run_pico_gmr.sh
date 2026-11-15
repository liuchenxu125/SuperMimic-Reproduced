#!/usr/bin/env bash
# Full-body PICO + ankle Trackers -> actual upstream GMR -> existing motion actor.
set -euo pipefail
SM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$SM_ROOT/scripts/teleop/run_pico.sh" --retargeter gmr --input-mode full-body "$@"
