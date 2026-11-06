#!/usr/bin/env bash
# Submit a sharded evaluation as a SLURM job array.
#   bash slurm/submit.sh DRIVER NODES [RUN] [extra sbatch args...]
# Each array task takes one node and runs one shard per GPU, so the scenes (the
# 1485 of navtest_full, or SCENE_LIST's) are split into NODES x GPUS_PER_NODE shards.
#
#   bash slurm/submit.sh drivor 4                                  # 16 GPUs
#   SHARD_LIMIT=2 bash slurm/submit.sh drivor 2 smoke --time=00:30:00
#   N_ROLLOUTS=3 bash slurm/submit.sh transfuser 4 ltf_x3
#   TRACK=pai SCENE_LIST=$PWD/configs/pai_scenes/nurec_curated_val.yaml bash slurm/submit.sh drivor 8
#
# Resubmitting with the same RUN (and NODES) reruns only unfinished shards. The
# merged summary is written to runs/<RUN>/metrics.txt when the last shard ends.
set -euo pipefail
HPC_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$HPC_ROOT/lib/common.sh"

DRIVER=${1:?usage: submit.sh DRIVER NODES [RUN] [sbatch args...]}
NODES=${2:?usage: submit.sh DRIVER NODES [RUN] [sbatch args...]}
RUN=${3:-$(basename "$DRIVER" .sh)_full_$(date +%m%d_%H%M)}
RUN_DIR=$(realpath -m "${RUNS_ROOT:-$HPC_ROOT/runs}/$RUN")
[ -f "$DRIVER" ] && DRIVER=$(realpath "$DRIVER")
# fail here rather than in every shard
[ "${TRACK:-nuplan}" != pai ] || ego_hood_overrides >/dev/null || exit 1

# the shard split depends on the number of shards; refuse to mix
if [ -f "$RUN_DIR/.num_shards" ] && [ "$(cat "$RUN_DIR/.num_shards")" != $((NODES * GPUS_PER_NODE)) ]; then
  echo "ERROR: $RUN_DIR was split into $(cat "$RUN_DIR/.num_shards") shards; use the same NODES or a new RUN" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"
echo $((NODES * GPUS_PER_NODE)) > "$RUN_DIR/.num_shards"

ACCOUNT=()
[ -z "$SLURM_ACCOUNT" ] || ACCOUNT=(--account="$SLURM_ACCOUNT")
sbatch "${ACCOUNT[@]}" --partition="$SLURM_PARTITION" \
  "${SLURM_RESOURCES[@]}" \
  --array="0-$((NODES - 1))" --job-name="alpasim_$RUN" \
  --output="$RUN_DIR/slurm-%A_%a.out" \
  --export="ALL,HPC_ROOT=$HPC_ROOT,DRIVER=$DRIVER,RUN_DIR=$RUN_DIR,GPUS_PER_NODE=$GPUS_PER_NODE" \
  "${@:4}" "$HPC_ROOT/slurm/job.sbatch"
echo ">> output: $RUN_DIR"
