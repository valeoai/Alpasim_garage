#!/usr/bin/env bash
# Closed-loop AlpaSim evaluation of one driver on one GPU, without Docker:
# the nuPlan (MTGS) track by default, or the PAI (NuRec) track with TRACK=pai
# (x86_64 only; its NRE renderer runs under Apptainer, see setup/pull_nre.sh).
#
#   bash run_eval.sh DRIVER [PRESET] [RUN] [wizard overrides...]
#
#   DRIVER  drivor | starter | transfuser | diffusiondrive | gtrs_dense,
#           or the path to your own driver script (see drivers/README.md)
#   PRESET  nuplan: dev (1 scene, with video; default) | full (1485 navtest scenes)
#           pai:    dev (AlpaSim's e2e_challenge/dev: 1 NuRec scene, with video;
#                   other scenes via scenes.scene_ids=[...] or scenes.test_suite_id=...)
#   RUN     output folder name under runs/ (default <driver>_<preset>_<date>)
#
# Examples:
#   bash run_eval.sh starter dev
#   bash run_eval.sh drivor dev one_scene "scenes.scene_ids=['<scene_id>']"
#   N_ROLLOUTS=3 bash run_eval.sh transfuser full
#
# Optional environment:
#   GPU=0             GPU for the renderer and the driver
#   BASEPORT=6000     renderer on BASEPORT, controller on BASEPORT+1
#   DRIVER_PORT=6789  driver gRPC port
#   N_ROLLOUTS=1      rollouts per scene
#   VIDEO=0|1         force video rendering off/on (default: dev on, full off)
#   RUNS_ROOT=runs    where run folders go
#   TRACK=nuplan|pai  challenge track (default nuplan)
#   WIZARD_OVERRIDES  extra wizard overrides, space-separated (also reaches SLURM shards)
set -euo pipefail
type module >/dev/null 2>&1 || [ "${ALPASIM_RELOGIN:-}" = "$0" ] || ALPASIM_RELOGIN=$0 exec bash -l "$0" "$@"
HPC_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HPC_ROOT/lib/common.sh"

DRIVER=${1:?usage: run_eval.sh DRIVER [dev|full] [RUN] [wizard overrides...]}
PRESET=${2:-dev}
RUN=${3:-$(basename "$DRIVER" .sh)_${PRESET}_$(date +%m%d_%H%M)}
OVERRIDES=("${@:4}")
GPU=${GPU:-0}
TRACK=${TRACK:-nuplan}
BASEPORT=${BASEPORT:-6000}
export ALPASIM_DRIVER_PORT=${DRIVER_PORT:-6789}
export ALPASIM_DRIVER_HOST=localhost
LOG_DIR=${LOG_DIR:-$(realpath -m "${RUNS_ROOT:-$HPC_ROOT/runs}/$RUN")}

[ -n "${N_ROLLOUTS:-}" ] && OVERRIDES+=("runtime.simulation_config.n_rollouts=$N_ROLLOUTS")
[ -n "${VIDEO:-}" ] && OVERRIDES+=("eval.video.render_video=$([ "$VIDEO" = 1 ] && echo true || echo false)")
if [ -n "${WIZARD_OVERRIDES:-}" ]; then read -ra _more <<<"$WIZARD_OVERRIDES"; OVERRIDES+=("${_more[@]}"); fi

load_modules
activate_env
# Inside a SLURM job array, AlpaSim would split the scenes again by array task
# and leave aggregation to the "last" task. Sharding is done by slurm/, so hide
# the array from AlpaSim.
for var in $(compgen -e | grep '^SLURM_ARRAY_' || true); do unset "$var"; done
mkdir -p "$LOG_DIR"
echo ">> driver $DRIVER, track $TRACK, preset $PRESET, GPU $GPU, ports $BASEPORT/$((BASEPORT + 1)) + driver :$ALPASIM_DRIVER_PORT"
echo ">> output: $LOG_DIR"

# 1. driver, in the background on this GPU
CUDA_VISIBLE_DEVICES=$GPU ALPASIM_DRIVER_LOG_DIR=$LOG_DIR/driver-logs \
  start_driver "$DRIVER" "$LOG_DIR/driver.log"
trap 'kill $DRIVER_PID 2>/dev/null || true' EXIT

# 2. wizard: write the service configs and docker-compose.yaml, start nothing
case $TRACK in
  nuplan) TRACK_ARGS=(+e2e_challenge_nuplan="$PRESET") ;;
  # NuRec scenes are fetched from Hugging Face into $ALPASIM_PAI_ROOT/nre-artifacts
  # (no internet on compute nodes: prefetch with setup/download_pai_scenes.sh)
  pai) TRACK_ARGS=(+e2e_challenge="$PRESET" "defines.filesystem=$ALPASIM_PAI_ROOT"
                   "services.physics.gpus=[$GPU]") ;;
  *) echo "ERROR: TRACK must be nuplan or pai, not '$TRACK'" >&2; exit 1 ;;
esac
# Run from $LOG_DIR, not the alpasim repo: the wizard and its check_config subprocess are Hydra
# apps that create outputs/<date>/ in their working directory, and the repo's filesystem may be
# out of inodes. Every config path the wizard writes is absolute, so the directory is free to pick.
(cd "$LOG_DIR" && alpasim_wizard "${TRACK_ARGS[@]}" \
  wizard.run_method=NONE wizard.log_dir="$LOG_DIR" wizard.baseport="$BASEPORT" \
  hydra.run.dir="$LOG_DIR/hydra" \
  "services.renderer.gpus=[$GPU]" "${OVERRIDES[@]}") > "$LOG_DIR/wizard.log" 2>&1 \
  || { echo "wizard failed, see $LOG_DIR/wizard.log"; tail -5 "$LOG_DIR/wizard.log"; exit 1; }

# 3. wait until the driver has loaded its model, then run renderer +
#    controller + runtime as host processes
python "$HPC_ROOT/tools/wait_for_driver.py" "localhost:$ALPASIM_DRIVER_PORT" --pid "$DRIVER_PID" \
  || { echo "see $LOG_DIR/driver.log"; tail -5 "$LOG_DIR/driver.log"; exit 1; }
python "$HPC_ROOT/lib/run_alpasim_local.py" "$LOG_DIR"

python "$HPC_ROOT/tools/summarize_run.py" "$LOG_DIR"
