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
