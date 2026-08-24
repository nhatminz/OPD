#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The underlying launcher is kept for backward compatibility, but now uses
# every visible GPU (or TRAIN_NPROC_PER_NODE when explicitly provided).
exec bash "${SCRIPT_DIR}/smoke_test_fsdp_2gpu.sh" "$@"
