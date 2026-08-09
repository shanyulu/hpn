#!/usr/bin/env bash
set -euo pipefail
FLUX_ROOT=${1:-/root/tencent-hpn-issue6/src/flux}
PATCH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$FLUX_ROOT"
git apply "$PATCH_DIR/flux_issue6.patch"
