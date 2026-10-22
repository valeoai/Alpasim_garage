#!/usr/bin/env bash
# Prefetch PAI-track NuRec scenes (.usdz, from the Hugging Face dataset
# nvidia/PhysicalAI-Autonomous-Vehicles-NuRec) into $ALPASIM_PAI_ROOT/nre-artifacts,
# so runs on compute nodes (no internet) find them. Run on a login node.
#   bash setup/download_pai_scenes.sh                      # the dev preset's scene
#   bash setup/download_pai_scenes.sh "scenes.scene_ids=['clipgt-...','clipgt-...']"
#   bash setup/download_pai_scenes.sh scenes.scene_ids=null scenes.test_suite_id=<suite>
# Scene IDs and suites: $ALPASIM_REPO/data/scenes/sim_scenes.csv, sim_suites.csv.
# It runs the wizard in config-only mode, which downloads whatever is missing.
set -euo pipefail
type module >/dev/null 2>&1 || [ "${ALPASIM_RELOGIN:-}" = "$0" ] || ALPASIM_RELOGIN=$0 exec bash -l "$0" "$@"
HPC_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$HPC_ROOT/lib/common.sh"
load_modules
activate_env

# the env's HF_HOME is the project cache; reuse the login's Hugging Face token
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
  HF_TOKEN=$(cat "$HOME/.cache/huggingface/token"); export HF_TOKEN
fi
log_dir=$(mktemp -d "${TMPDIR:-/tmp}/pai-scenes.XXXXXX")
trap 'rm -rf "$log_dir"' EXIT
(cd "$ALPASIM_REPO" && alpasim_wizard +e2e_challenge=dev wizard.run_method=NONE \
  wizard.log_dir="$log_dir" "defines.filesystem=$ALPASIM_PAI_ROOT" "$@")
echo ">> scenes cached in $ALPASIM_PAI_ROOT/nre-artifacts/all-usdzs:"
find "$ALPASIM_PAI_ROOT/nre-artifacts/all-usdzs" -name "*.usdz" | wc -l
