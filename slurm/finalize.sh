#!/usr/bin/env bash
# Merge and score a sharded run once all its shards have finished.
#   bash slurm/finalize.sh RUN_DIR
# Called by every array task at its end; only the task that sees all shards
# done writes the summary (runs/<RUN>/metrics.txt, metrics.json). Safe to rerun.
set -euo pipefail
type module >/dev/null 2>&1 || [ "${ALPASIM_RELOGIN:-}" = "$0" ] || ALPASIM_RELOGIN=$0 exec bash -l "$0" "$@"
HPC_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$HPC_ROOT/lib/common.sh"

RUN_DIR=$1
expected=$(cat "$RUN_DIR/.num_shards")
done_count=$(ls "$RUN_DIR"/shard_*/aggregate/results-summary.json 2>/dev/null | wc -l)
if [ "$done_count" -lt "$expected" ]; then
  echo ">> $done_count/$expected shards finished; the last task to finish writes the summary"
  exit 0
fi
load_modules
activate_env
python "$HPC_ROOT/tools/summarize_run.py" "$RUN_DIR" --scenes "${SCENE_LIST:-navtest_full}"
