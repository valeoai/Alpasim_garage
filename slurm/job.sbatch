#!/bin/bash
# One node of a sharded full-navtest evaluation: runs GPUS_PER_NODE shards,
# one per GPU. Submit with slurm/submit.sh, which sets the account, partition,
# array size and these variables:
#   DRIVER, RUN_DIR, GPUS_PER_NODE, [SHARD_LIMIT, N_ROLLOUTS]
#SBATCH --job-name=alpasim
#SBATCH --nodes=1
set -eo pipefail
HPC_ROOT=${HPC_ROOT:?submit with slurm/submit.sh}
TASKS=${SLURM_ARRAY_TASK_COUNT:-1}
TASK=${SLURM_ARRAY_TASK_ID:-0}
# the run's shard count (submit.sh writes it), not this array's size: resubmitting only some
# tasks (sbatch --array=4,11,58-110) must keep every shard on the same scenes
NUM_SHARDS=$(cat "$RUN_DIR/.num_shards" 2>/dev/null || echo $((GPUS_PER_NODE * TASKS)))
mkdir -p "$RUN_DIR"
if [ $(((TASK + 1) * GPUS_PER_NODE)) -gt "$NUM_SHARDS" ]; then
  echo "ERROR: array task $TASK is past the run's $NUM_SHARDS shards (tasks 0-$((NUM_SHARDS / GPUS_PER_NODE - 1)))" >&2
  exit 1
fi

# ports by the node's physical GPU id (SLURM_JOB_GPUS, e.g. 4,5,6,7), not the job's local index: two half-node
# jobs on one node both see GPUs 0-3, and with ports from those they bind the same ones and talk to each other
IFS=, read -ra PHYS_GPUS <<< "${SLURM_JOB_GPUS:-${SLURM_STEP_GPUS:-}}"
pids=()
for GPU in $(seq 0 $((GPUS_PER_NODE - 1))); do
  SHARD=$((TASK * GPUS_PER_NODE + GPU))
  PORT_SLOT=${PHYS_GPUS[$GPU]:-$GPU} bash -l "$HPC_ROOT/slurm/run_shard.sh" "$DRIVER" "$SHARD" "$NUM_SHARDS" "$GPU" "$RUN_DIR" \
    > "$RUN_DIR/$(printf "shard_%03d" "$SHARD").out" 2>&1 &
  pids+=($!)
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
bash -l "$HPC_ROOT/slurm/finalize.sh" "$RUN_DIR" > "$RUN_DIR/finalize_${TASK}.out" 2>&1 || true
exit $rc
