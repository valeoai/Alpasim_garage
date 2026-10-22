#!/usr/bin/env bash
# AlpaSim's starter driver: ignores the cameras and drives straight ahead at
# the current speed. No model or GPU; use it to test the pipeline.
set -euo pipefail
type module >/dev/null 2>&1 || [ "${ALPASIM_RELOGIN:-}" = "$0" ] || ALPASIM_RELOGIN=$0 exec bash -l "$0" "$@"
HPC_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$HPC_ROOT/lib/common.sh"
driver_setup

exec python "$ALPASIM_REPO/e2e_challenge/starter_kit/driver.py"
