#!/usr/bin/env bash
# AlpaSim's SimScale DiffusionDrive sample driver, run from its copy in
# drivers/simscale/diffusiondrive instead of its Docker image.
# Checkpoint: bash setup/download_checkpoints.sh diffusiondrive
set -euo pipefail
type module >/dev/null 2>&1 || [ "${ALPASIM_RELOGIN:-}" = "$0" ] || ALPASIM_RELOGIN=$0 exec bash -l "$0" "$@"
HPC_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$HPC_ROOT/lib/common.sh"
driver_setup
# PAI track: nuPlan camera views from the PAI rig (drivor_alpasim.pai_views)
export PYTHONPATH=$HPC_ROOT/drivers/drivor${PYTHONPATH:+:$PYTHONPATH}

export DIFFUSIONDRIVE_CHECKPOINT_PATH=${DIFFUSIONDRIVE_CHECKPOINT_PATH:-$CKPT_ROOT/simscale/SimScale_ckpts/DiffusionDrive/diffusiondrive_sim_navhard.ckpt}
export DIFFUSIONDRIVE_MAX_BATCH_SIZE=${DIFFUSIONDRIVE_MAX_BATCH_SIZE:-$DRIVER_MAX_BATCH_SIZE}
cd "$HPC_ROOT/drivers/simscale/diffusiondrive"
exec python -m navsim_diffusiondrive_challenge.driver
