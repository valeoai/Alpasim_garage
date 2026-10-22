#!/usr/bin/env bash
# Download the public checkpoints of the baseline drivers into $CKPT_ROOT (ckpts/ by default).
#   bash setup/download_checkpoints.sh [drivor] [transfuser] [diffusiondrive] [gtrs_dense] [harmonizer]
# With no arguments, downloads all of them (~3.6 GB). Run on a login node (needs internet).
#
#   drivor          vanilla DrivoR, from its GitHub release                -> drivor_vanilla/
#   transfuser      SimScale Latent TransFuser  (OpenDriveLab/SimScale, HF) -> simscale/SimScale_ckpts/
#   diffusiondrive  SimScale DiffusionDrive     (OpenDriveLab/SimScale, HF) -> simscale/SimScale_ckpts/
#   gtrs_dense      SimScale GTRS-Dense         (OpenDriveLab/SimScale, HF) -> simscale/SimScale_ckpts/
#   harmonizer      the PAI renderer's harmonizer (nvidia/DiffusionHarmonizer, HF) -> nvidia_renderer_ckpt/
set -euo pipefail
HPC_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$HPC_ROOT/lib/common.sh"
activate_env   # for the `hf` CLI
export HF_HOME=$CACHE_ROOT/huggingface

DRIVOR_URL=https://github.com/valeoai/DrivoR/releases/download/model_weights/drivor_Nav1_25epochs.pth
DRIVOR_SHA256=e1a678f201e4f1ab93d117caad42782cd7ead293bdced2b5f80212bc92426ae3
declare -A SIMSCALE=(
  [transfuser]=SimScale_ckpts/LTF/ltf_sim_navtest.ckpt
  [diffusiondrive]=SimScale_ckpts/DiffusionDrive/diffusiondrive_sim_navhard.ckpt
  [gtrs_dense]=SimScale_ckpts/GTRS_Dense/gtrs_dense_resnet_sim_reward_navhard.ckpt
)
WANT=("$@")
[ ${#WANT[@]} -eq 0 ] && WANT=(drivor transfuser diffusiondrive gtrs_dense harmonizer)

for name in "${WANT[@]}"; do
  case $name in
    drivor)
      out=$CKPT_ROOT/drivor_vanilla/drivor_Nav1_25epochs.pth
      echo ">> drivor: $DRIVOR_URL"
      mkdir -p "$(dirname "$out")"
      [ -f "$out" ] || { curl -sSfL -o "$out.part" "$DRIVOR_URL" && mv "$out.part" "$out"; }
      echo "$DRIVOR_SHA256  $out" | sha256sum -c
      ;;
    transfuser|diffusiondrive|gtrs_dense)
      echo ">> $name: OpenDriveLab/SimScale ${SIMSCALE[$name]}"
      hf download OpenDriveLab/SimScale "${SIMSCALE[$name]}" --repo-type dataset --local-dir "$CKPT_ROOT/simscale"
      ;;
    harmonizer)
      echo ">> harmonizer: nvidia/DiffusionHarmonizer harmonizer_nontemporal.pt (PAI track only)"
      hf download nvidia/DiffusionHarmonizer harmonizer_nontemporal.pt --local-dir "$NRE_HARMONIZER_DIR"
      ;;
    *) echo "ERROR: unknown checkpoint '$name' (drivor|transfuser|diffusiondrive|gtrs_dense|harmonizer)" >&2; exit 1 ;;
  esac
done
echo ">> checkpoints in $CKPT_ROOT:"
find "$CKPT_ROOT" -type f \( -name "*.pth" -o -name "*.ckpt" -o -name "*.pt" \) -exec ls -lh {} \; | awk '{print "   " $5 "  " $NF}'
