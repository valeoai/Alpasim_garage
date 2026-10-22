#!/usr/bin/env bash
# Vanilla DrivoR (valeoai/DrivoR, 4-camera NAVSIM model) as an AlpaSim driver.
#   DRIVOR_CHECKPOINT_PATH=/path/drivor.pth bash drivers/drivor.sh
# Checkpoint: bash setup/download_checkpoints.sh drivor
# The network is in drivers/drivor/drivor_alpasim/original (only the model is imported).
set -euo pipefail
type module >/dev/null 2>&1 || [ "${ALPASIM_RELOGIN:-}" = "$0" ] || ALPASIM_RELOGIN=$0 exec bash -l "$0" "$@"
HPC_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$HPC_ROOT/lib/common.sh"
driver_setup

export DRIVOR_CHECKPOINT_PATH
export DRIVOR_MAX_BATCH_SIZE=${DRIVOR_MAX_BATCH_SIZE:-$DRIVER_MAX_BATCH_SIZE}
cd "$HPC_ROOT/drivers/drivor"
exec python -m drivor_alpasim.driver
