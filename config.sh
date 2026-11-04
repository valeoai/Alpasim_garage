# Site configuration, sourced by every script in this repo.
# Edit the block below for your cluster, or override any variable by exporting it
# before calling a script (e.g. `PROJECT=/scratch/me/alpasim bash setup/create_env.sh`).

# ============================ edit for your cluster ============================

# Base directory for code, data, envs and caches. Use project/scratch storage, not
# $HOME: the nuPlan-track data alone is ~560 GB and ~115k files.
: "${PROJECT:=}"

# Environment modules that provide CUDA 12.x, GCC (>= 11) and Ninja, loaded before
# anything runs; gsplat JIT-compiles its CUDA kernels against them. Leave empty if
# they are already on PATH (or your cluster has no module system). Example:
#   ALPASIM_MODULES=(GCC/13.3.0 Ninja/1.12.1 CUDA/12.9.0)
[ -n "${ALPASIM_MODULES+x}" ] || ALPASIM_MODULES=()
# Extra modules needed only while building the env: Rust >= 1.74 (for AlpaSim's
# utils_rs). Leave empty if `cargo` is on PATH (e.g. from rustup). Example:
#   ALPASIM_BUILD_MODULES=(Rust/1.78.0)
[ -n "${ALPASIM_BUILD_MODULES+x}" ] || ALPASIM_BUILD_MODULES=()

# SLURM (slurm/submit.sh): account (empty: none passed), GPU partition, GPUs per node,
# and the per-node request. One array task takes a whole node and runs one pipeline
# per GPU, so ask for all its GPUs and CPUs.
: "${SLURM_ACCOUNT:=}"
: "${SLURM_PARTITION:=gpu}"
: "${GPUS_PER_NODE:=4}"
[ -n "${SLURM_RESOURCES+x}" ] || SLURM_RESOURCES=(--ntasks-per-node=1 --gpus-per-node="$GPUS_PER_NODE" --exclusive --time=03:00:00)

# ===============================================================================

# The three lists above can also be exported as space-separated strings (bash can't
# export arrays), e.g. ALPASIM_MODULES="GCC/13.3.0 CUDA/12.9.0"; split them here.
read -ra ALPASIM_MODULES <<< "${ALPASIM_MODULES[*]}"
read -ra ALPASIM_BUILD_MODULES <<< "${ALPASIM_BUILD_MODULES[*]}"
read -ra SLURM_RESOURCES <<< "${SLURM_RESOURCES[*]}"

if [ -z "$PROJECT" ]; then
  echo "ERROR: set PROJECT in config.sh (or export it): the base directory for code, data, envs and caches" >&2
  return 1 2>/dev/null || exit 1
fi

# This repo; its ckpts/ folder (git-ignored) holds the driver checkpoints.
: "${HPC_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
: "${CKPT_ROOT:=$HPC_ROOT/ckpts}"

# AlpaSim checkout (e2e_challenge branch); setup/create_env.sh clones it here.
: "${ALPASIM_REPO:=$PROJECT/code/alpasim}"
: "${ALPASIM_BRANCH:=e2e_challenge}"
# Commit to check out (setup/create_env.sh); set it empty to keep the branch head. The default
# e121e59 includes AlpaSim's lateral-corridor fix (1b024f0, README section 10), with which the
# README's baseline results are scored. drivers/simscale/ comes from 2023534, two commits earlier
# (the commits in between change only the scorer and the challenge CLI); ALPASIM_COMMIT=2023534
# reproduces the results scored before the fix.
: "${ALPASIM_COMMIT=e121e59}"

# nuPlan-track data; setup/download_nuplan_track.sh fills it (~560 GB).
: "${ALPASIM_NUPLAN_ROOT:=$PROJECT/alpasim-nuplan-track}"

# PAI (NuRec) track, x86_64 only. The closed-source NRE renderer runs from its NGC
# image under Apptainer/Singularity (setup/pull_nre.sh builds the .sif); NuRec scenes
# (.usdz) are cached under $ALPASIM_PAI_ROOT/nre-artifacts (setup/download_pai_scenes.sh).
: "${NRE_IMAGE:=nvcr.io/nvidia/nre/nre-ga:26.04}"
: "${NRE_SIF:=$PROJECT/images/$(basename "${NRE_IMAGE//:/_}").sif}"
: "${ALPASIM_PAI_ROOT:=$PROJECT/alpasim-pai}"
: "${APPTAINER_CACHEDIR:=$PROJECT/.cache/apptainer}"
# Folder holding harmonizer_nontemporal.pt (the official config enables NRE's
# harmonizer, which otherwise downloads it from Hugging Face; compute nodes have no internet).
: "${NRE_HARMONIZER_DIR:=$CKPT_ROOT/nvidia_renderer_ckpt}"
export NRE_IMAGE NRE_SIF ALPASIM_PAI_ROOT APPTAINER_CACHEDIR NRE_HARMONIZER_DIR

# Caches (pip, uv, cargo, torch extensions, Hugging Face), kept off $HOME.
: "${CACHE_ROOT:=$PROJECT/.cache}"

# Env manager: micromamba (default) or conda (then CONDA_ROOT: the conda install).
# ALPASIM_ENV is an env name (under $MAMBA_ROOT_PREFIX/envs/) or a path (prefix env).
: "${ENV_MANAGER:=micromamba}"
[ "$ENV_MANAGER" = conda ] && : "${MAMBA_ROOT_PREFIX:=${CONDA_ROOT:?set CONDA_ROOT for ENV_MANAGER=conda}}"
: "${MAMBA_ROOT_PREFIX:=$PROJECT/env/micromamba}"
# Falls back to the micromamba found on PATH.
: "${MAMBA_EXE:=$(type -P micromamba || echo "$PROJECT/env/.local/bin/micromamba")}"
: "${ALPASIM_ENV:=alpasim}"

# DrivoR driver (drivers/drivor.sh): the vanilla DrivoR checkpoint
# (setup/download_checkpoints.sh drivor).
: "${DRIVOR_CHECKPOINT_PATH:=$CKPT_ROOT/drivor_vanilla/drivor_Nav1_25epochs.pth}"
