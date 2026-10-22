#!/usr/bin/env bash
# Download and extract the AlpaSim nuPlan-track data from Hugging Face
# (OpenDriveLab/AlpasimChallenge2026_nuplan_track, public, no token needed).
#
#   bash setup/download_nuplan_track.sh          # full navtest set: 17 files, ~491 GB
#   bash setup/download_nuplan_track.sh dev      # dev smoke test only: 3 files, ~36 GB
#
# Run on a LOGIN node (compute nodes have no internet), inside tmux or with
#   nohup bash setup/download_nuplan_track.sh > download.log 2>&1 &
# One tarball at a time: download -> extract -> delete, so the extra disk
# needed at peak is one shard (~35 GB). Re-running resumes: finished files are
# recorded in $ALPASIM_NUPLAN_ROOT/.done/.
set -euo pipefail
HPC_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$HPC_ROOT/lib/common.sh"
activate_env   # for the `hf` CLI

WHAT=${1:-full}
REPO=OpenDriveLab/AlpasimChallenge2026_nuplan_track
ROOT=$ALPASIM_NUPLAN_ROOT
DL=$ROOT/_tarballs
DONE=$ROOT/.done
export HF_HOME=$CACHE_ROOT/huggingface

FILES=(trajdata_cache/nuplan_test.tar.gz MTGS_asset/navtest/configs.tar.gz)
case $WHAT in
  dev)  FILES+=(MTGS_asset/navtest/assets/part001.tar.gz) ;;
  full) for i in $(seq -w 1 15); do FILES+=("MTGS_asset/navtest/assets/part0$i.tar.gz"); done ;;
  *)    echo "usage: $0 [full|dev]" >&2; exit 2 ;;
esac

mkdir -p "$ROOT" "$DL" "$DONE"
for f in "${FILES[@]}"; do
  tag=$(echo "$f" | tr / _)
  if [ -e "$DONE/$tag" ]; then echo ">> skip $f (done)"; continue; fi

  echo ">> [$(date +%T)] download $f"
  hf download "$REPO" "$f" --repo-type dataset --local-dir "$DL"

  echo ">> [$(date +%T)] extract  $f"
  tar xzf "$DL/$f" -C "$ROOT"      # each tarball carries its own navtest/... or nuplan_test/

  rm -f "$DL/$f"
  touch "$DONE/$tag"
  echo ">> [$(date +%T)] ok       $f   (free: $(df -h --output=avail "$ROOT" | tail -1))"
done

rm -rf "$DL"
echo ">> all done"
echo "   scenes with assets: $(ls "$ROOT/navtest/assets" | wc -l)"
echo "   scene configs:      $(ls "$ROOT/navtest/configs" | wc -l)"
echo "   trajdata scenes:    $(ls "$ROOT/nuplan_test" | wc -l)"
