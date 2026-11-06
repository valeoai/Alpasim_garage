# Shared shell helpers. Source after setting HPC_ROOT to the repo root:
#   HPC_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); source "$HPC_ROOT/lib/common.sh"

# shellcheck source=../config.sh
source "$HPC_ROOT/config.sh" || exit 1
export PROJECT ALPASIM_REPO ALPASIM_NUPLAN_ROOT CKPT_ROOT CACHE_ROOT MAMBA_ROOT_PREFIX MAMBA_EXE ENV_MANAGER CONDA_ROOT

# `module` is a shell function defined by the login profile. Entry scripts
# re-exec themselves once under `bash -l` when it is missing (batch jobs, clean envs):
#   type module >/dev/null 2>&1 || [ "${ALPASIM_RELOGIN:-}" = "$0" ] || ALPASIM_RELOGIN=$0 exec bash -l "$0" "$@"
# On a machine without environment modules (e.g. a cloud VM), CUDA, GCC and Ninja
# must already be on PATH, and the modules are skipped.
load_modules() {
  if ! type module >/dev/null 2>&1; then
    echo "WARNING: no 'module' command; skipping modules, using the CUDA/GCC/Ninja on PATH" >&2
    return 0
  fi
  local m
  for m in "${ALPASIM_MODULES[@]}" "$@"; do module load "$m"; done
  return 0
}

# Define the `micromamba` or `conda` shell function ($ENV_MANAGER).
env_hook() {
  if [ "$ENV_MANAGER" = conda ]; then
    # shellcheck disable=SC1091
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
  else
    eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
  fi
}

activate_env() {
  env_hook
  "$ENV_MANAGER" activate "$ALPASIM_ENV"
  # The wizard imports GitPython, which aborts if `git` is missing (as on some
  # compute nodes). git is only used for cloning driver code, which this setup
  # never does.
  export GIT_PYTHON_REFRESH=quiet
}

# PAI track: the ego-car hood. As in the challenge, NRE draws the hood of the recording car
# over every camera_front_wide_120fov frame (runtime.simulation_config.ego_mask_rig_config_id
# = hyperion_8_1, AlpaSim's default). The hood images ship in the AlpaSim repo
# (data/nre-artifacts/ego-hoods/<rig>/<camera>.png, ordinary files, not on Hugging Face); the
# renderer reads them from <data root>/nre-artifacts/ego-hoods, here $ALPASIM_PAI_ROOT's. If
# that folder is missing, the renderer gets an empty one and silently renders no hood.
# EGO_HOOD=0 turns the hood off (ego_mask_rig_config_id=null), as in the runs scored before
# it was enabled.
EGO_HOODS_DIR=$ALPASIM_PAI_ROOT/nre-artifacts/ego-hoods
EGO_HOOD_RIG=hyperion_8_1
EGO_HOOD_FILE=$EGO_HOODS_DIR/$EGO_HOOD_RIG/camera_front_wide_120fov.png

# Copy the hood images from the AlpaSim checkout into $EGO_HOODS_DIR.
install_ego_hoods() {
  local src=$ALPASIM_REPO/data/nre-artifacts/ego-hoods
  if [ ! -f "$src/$EGO_HOOD_RIG/camera_front_wide_120fov.png" ]; then
    echo "ERROR: no ego hood in $src; is ALPASIM_REPO an AlpaSim e2e_challenge checkout?" >&2
    return 1
  fi
  mkdir -p "$EGO_HOODS_DIR"
  cp -r "$src/." "$EGO_HOODS_DIR/"
  echo ">> ego hoods in $EGO_HOODS_DIR:"
  (cd "$EGO_HOODS_DIR" && find . -name "*.png" | sort)
}

# Wizard overrides for the hood, or an error if it is on but not installed.
ego_hood_overrides() {
  case ${EGO_HOOD:-1} in
    1) if [ ! -f "$EGO_HOOD_FILE" ]; then
         echo "ERROR: the PAI ego hood is missing ($EGO_HOOD_FILE). Install it with" \
              "'bash setup/download_pai_scenes.sh' (it copies it from the AlpaSim checkout)," \
              "or set EGO_HOOD=0 to run without it." >&2
         return 1
       fi ;;
    0) echo "runtime.simulation_config.ego_mask_rig_config_id=null" ;;
    *) echo "ERROR: EGO_HOOD must be 1 or 0, not '$EGO_HOOD'" >&2; return 1 ;;
  esac
}

# Start driver script $1 (a name from drivers/ or a path) in the background.
# Sets DRIVER_PID. Output goes to $2.
start_driver() {
  local driver=$1 log=$2 script
  if [ -f "$driver" ]; then script=$driver; else script=$HPC_ROOT/drivers/$driver.sh; fi
  [ -f "$script" ] || { echo "ERROR: unknown driver '$driver' (no $script)" >&2; exit 1; }
  bash "$script" > "$log" 2>&1 &
  DRIVER_PID=$!
}

# For drivers/*.sh: load modules + env unless the caller already did, and set
# the variables every driver shares. A driver listens on
# $ALPASIM_DRIVER_HOST:$ALPASIM_DRIVER_PORT and runs on $CUDA_VISIBLE_DEVICES.
driver_setup() {
  if [ "${CONDA_DEFAULT_ENV:-}" != "$ALPASIM_ENV" ]; then
    load_modules
    activate_env
  fi
  export ALPASIM_DRIVER_HOST=${ALPASIM_DRIVER_HOST:-0.0.0.0}
  export ALPASIM_DRIVER_PORT=${ALPASIM_DRIVER_PORT:-6789}
  export ALPASIM_DRIVER_LOG_DIR=${ALPASIM_DRIVER_LOG_DIR:-${TMPDIR:-/tmp}/alpasim-driver-$ALPASIM_DRIVER_PORT}
  export TORCH_NUM_THREADS=${TORCH_NUM_THREADS:-4}
  export HF_HUB_OFFLINE=1   # compute nodes have no internet; all weights are local
  DRIVER_MAX_BATCH_SIZE=${DRIVER_MAX_BATCH_SIZE:-4}
}
