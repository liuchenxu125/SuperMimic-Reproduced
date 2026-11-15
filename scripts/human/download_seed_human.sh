#!/usr/bin/env bash
# HF 2.x: explicit filenames; use the tested local HTTP proxy, not socks://.
set -euo pipefail
SM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SM_HF_PROXY="${SM_HF_PROXY:-http://127.0.0.1:7897}"
SM_SEED_DIR="${SM_SEED_DIR:-/media/casbot/Linux_Drive/supermimic_ws/bones_seed_human}"
SM_ENV=(env -u PYTHONPATH -u ALL_PROXY -u all_proxy
    "HTTP_PROXY=$SM_HF_PROXY" "HTTPS_PROXY=$SM_HF_PROXY"
    "http_proxy=$SM_HF_PROXY" "https_proxy=$SM_HF_PROXY")
case "${1:-download}" in
    login)
        exec "${SM_ENV[@]}" "$SM_ROOT/.venv/bin/hf" auth login
        ;;
    download|plan)
        SM_ARGS=()
        if [[ "${1:-download}" == plan ]]; then SM_ARGS+=(--dry-run); fi
        exec "${SM_ENV[@]}" "$SM_ROOT/.venv/bin/hf" download bones-studio/seed \
            soma_uniform.tar.gz metadata/seed_metadata_v004.csv \
            soma_shapes/soma_base_rig/soma_base_skel_minimal.bvh \
            --repo-type dataset --local-dir "$SM_SEED_DIR" "${SM_ARGS[@]}"
        ;;
    *)
        echo "Usage: bash $0 [login|download|plan]" >&2
        exit 2
        ;;
esac
