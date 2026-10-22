#!/usr/bin/env bash
# AlpaSim's SimScale GTRS-Dense (ResNet, reward) sample driver, run from its copy in
# drivers/simscale/gtrs_dense instead of its Docker image.
# Checkpoint: bash setup/download_checkpoints.sh gtrs_dense
set -euo pipefail
type module >/dev/null 2>&1 || [ "${ALPASIM_RELOGIN:-}" = "$0" ] || ALPASIM_RELOGIN=$0 exec bash -l "$0" "$@"
HPC_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$HPC_ROOT/lib/common.sh"
driver_setup
# PAI track: nuPlan camera views from the PAI rig (drivor_alpasim.pai_views)
export PYTHONPATH=$HPC_ROOT/drivers/drivor${PYTHONPATH:+:$PYTHONPATH}

SAMPLE=$HPC_ROOT/drivers/simscale/gtrs_dense
export GTRS_CHECKPOINT_PATH=${GTRS_CHECKPOINT_PATH:-$CKPT_ROOT/simscale/SimScale_ckpts/GTRS_Dense/gtrs_dense_resnet_sim_reward_navhard.ckpt}
export GTRS_VOCAB_PATH=${GTRS_VOCAB_PATH:-$SAMPLE/assets/navsim_16384.npy}
export GTRS_MAX_BATCH_SIZE=${GTRS_MAX_BATCH_SIZE:-$DRIVER_MAX_BATCH_SIZE}
cd "$SAMPLE"
exec python -m navsim_gtrs_dense_challenge.driver
