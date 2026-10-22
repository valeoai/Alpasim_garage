#!/usr/bin/env bash
# One GPU's share of a sharded full-navtest evaluation (called by job.sbatch).
#   bash slurm/run_shard.sh DRIVER SHARD NUM_SHARDS GPU OUT_ROOT
# Runs scenes SHARD, SHARD+NUM_SHARDS, ... of navtest_full (sorted) on GPU $GPU,
# with ports offset by GPU so 4 shards can share a node. Output goes to
# $OUT_ROOT/shard_XXX/. A shard that already has results is skipped.
# SHARD_LIMIT=N keeps only the first N scenes of the shard (smoke tests).
# SCENE_LIST=file.yaml takes the scene ids from that file (scenes.scene_ids) instead of navtest_full.
# TRACK=pai runs the PAI (NuRec) track; give its scenes with SCENE_LIST (e.g. configs/pai_scenes/nurec_curated_val.yaml).
set -euo pipefail
type module >/dev/null 2>&1 || [ "${ALPASIM_RELOGIN:-}" = "$0" ] || ALPASIM_RELOGIN=$0 exec bash -l "$0" "$@"
HPC_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$HPC_ROOT/lib/common.sh"

DRIVER=$1; SHARD=$2; NUM_SHARDS=$3; GPU=$4; OUT_ROOT=$5
LOG_DIR=$OUT_ROOT/$(printf "shard_%03d" "$SHARD")

load_modules
activate_env
if [ -f "$LOG_DIR/aggregate/results-summary.json" ]; then
  echo ">> shard $SHARD already done, skipping"; exit 0
fi
rm -rf "$LOG_DIR"
SCENES=$(python - "$SHARD" "$NUM_SHARDS" "${SHARD_LIMIT:-0}" <<'PY'
import os, sys, yaml
shard, n, limit = map(int, sys.argv[1:])
path = os.environ.get("SCENE_LIST") or os.path.join(os.environ["ALPASIM_REPO"], "src/wizard/configs/nuplan_scenes/navtest_full.yaml")
ids = sorted(yaml.safe_load(open(path))["scenes"]["scene_ids"])[shard::n]
if limit > 0:
    ids = ids[:limit]
print("[" + ",".join(f"'{s}'" for s in ids) + "]")
PY
)
echo ">> shard $SHARD/$NUM_SHARDS: $(grep -o "'" <<<"$SCENES" | wc -l | awk '{print $1/2}') scenes"

# TRACK=pai: AlpaSim's PAI preset (dev) on the given scenes (SCENE_LIST), no videos by default
# (and both NuRec catalogs, 26.01 and 26.04, as AlpaSim's nurec_scenes configs load them)
PRESET=full
EXTRA=()
if [ "${TRACK:-nuplan}" = pai ]; then
  PRESET=dev; export VIDEO=${VIDEO:-0}
  EXTRA=("scenes.scenes_csv=[$ALPASIM_REPO/data/scenes/sim_scenes.csv,$ALPASIM_REPO/data/scenes/sim_scenes_2604.csv]")
fi
# ports by PORT_SLOT (job.sbatch: the physical GPU id), so shards of jobs sharing a node never meet; a port
# already listening means another shard owns it: stop rather than drive through its services
SLOT=${PORT_SLOT:-$GPU}
for port in $((6000 + 100 * SLOT)) $((6001 + 100 * SLOT)) $((6002 + 100 * SLOT)) $((7000 + SLOT)); do
  if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
    echo "ERROR: port $port is already in use on $(hostname) (another shard?); not starting" >&2
    exit 1
  fi
done
LOG_DIR=$LOG_DIR GPU=$GPU BASEPORT=$((6000 + 100 * SLOT)) DRIVER_PORT=$((7000 + SLOT)) \
  bash "$HPC_ROOT/run_eval.sh" "$DRIVER" "$PRESET" - "scenes.scene_ids=$SCENES" "${EXTRA[@]}"
