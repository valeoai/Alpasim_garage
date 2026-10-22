#!/usr/bin/env bash
# Build the container-free AlpaSim env (nuPlan/MTGS track) as a micromamba or
# conda env (ENV_MANAGER in config.sh).
#   bash setup/create_env.sh
# Upstream builds this inside a Docker image. Here the Dockerfile's steps run
# in a micromamba env instead, with CUDA/GCC/Ninja/Rust from environment
# modules. Run on a login node (needs internet). Takes ~15 min.
set -euo pipefail
type module >/dev/null 2>&1 || [ "${ALPASIM_RELOGIN:-}" = "$0" ] || ALPASIM_RELOGIN=$0 exec bash -l "$0" "$@"
HPC_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$HPC_ROOT/lib/common.sh"

load_modules "${ALPASIM_BUILD_MODULES[@]}"

# caches off $HOME (inode quota)
export PIP_CACHE_DIR=$CACHE_ROOT/pip
export UV_CACHE_DIR=$CACHE_ROOT/uv
export CARGO_HOME=$CACHE_ROOT/cargo
export CONDA_PKGS_DIRS=$CACHE_ROOT/conda-pkgs   # conda only; not the shared conda root
export TMPDIR=${TMPDIR:-$CACHE_ROOT/tmp}
mkdir -p "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$CARGO_HOME" "$CONDA_PKGS_DIRS" "$TMPDIR"

env_hook

# 1. env: python + uv from conda-forge; ffmpeg too (for rollout videos)
echo ">> creating env $ALPASIM_ENV"
case $ALPASIM_ENV in */*) env_flag=-p ;; *) env_flag=-n ;; esac
"$ENV_MANAGER" create -y "$env_flag" "$ALPASIM_ENV" -c conda-forge --override-channels python=3.12 uv ffmpeg
"$ENV_MANAGER" activate "$ALPASIM_ENV"
python -VV

# 2. source
# AlpaSim's LFS files (test logs, sample assets, reference results) aren't needed here. When
# the git config has LFS filters but no git-lfs binary (as on some clusters), a checkout would
# fail on them, so check out their pointer files instead.
lfs_off=()
git lfs version >/dev/null 2>&1 \
  || lfs_off=(-c filter.lfs.process= -c filter.lfs.smudge= -c filter.lfs.required=false)
if [ ! -d "$ALPASIM_REPO/.git" ]; then
  git "${lfs_off[@]}" clone -b "$ALPASIM_BRANCH" https://github.com/NVlabs/alpasim.git "$ALPASIM_REPO"
fi
cd "$ALPASIM_REPO"
# ALPASIM_COMMIT pins the checkout, e.g. to the commit a reported result used
[ -z "${ALPASIM_COMMIT:-}" ] || git "${lfs_off[@]}" checkout -q "$ALPASIM_COMMIT"
echo ">> AlpaSim at $(git log --oneline -1)"

# 3. protos (same as the Dockerfile). uv syncs into this conda env instead of a
# .venv; pyproject says python-preference = only-managed, so accept the env's
# python instead. --inexact keeps conda's own packages.
export UV_PROJECT_ENVIRONMENT=$CONDA_PREFIX
export UV_PYTHON=$CONDA_PREFIX/bin/python
export UV_PYTHON_PREFERENCE=only-system
export UV_LINK_MODE=copy
(cd src/grpc && uv sync --inexact && uv run --no-sync compile-protos)

# 4. alpasim + MTGS renderer plugin. pyqt5 (pulled in by alpasim-tools, only
# used for visualisation) has no aarch64 wheel, so skip it.
uv sync --inexact --extra all --extra mtgs \
  --no-install-package pyqt5 --no-install-package pyqt5-qt5 --no-install-package pyqt5-sip

# 5. CUDA torch. uv.lock pins torch 2.8.0 from PyPI, which is CPU-only on
# aarch64; the aarch64 CUDA build of 2.8.0 is on the cu129 index. Its CUDA
# major matches the CUDA/12 module, which torch checks when gsplat JIT-compiles.
# On x86_64 this step is harmless. --no-config: alpasim's pyproject sets
# exclude-newer, and the PyTorch index has no upload dates, so uv would
# filter out every wheel there.
uv pip install --no-config --reinstall-package torch --reinstall-package torchvision \
  "torch==2.8.0" "torchvision==0.23.0" --index-url https://download.pytorch.org/whl/cu129

# 6. extra driver dependencies (DrivoR image backbone)
uv pip install "timm==1.0.15"

# 7. activation hook. UV_NO_SYNC: a plain `uv run` would otherwise re-sync and
# put CPU torch back. TORCH_CUDA_ARCH_LIST (the GPU archs gsplat JIT-compiles for)
# defaults to this node's GPU, else 9.0 (H100/GH200).
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/alpasim.sh" <<EOF
: "\${TORCH_CUDA_ARCH_LIST:=\$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1)}"
export TORCH_CUDA_ARCH_LIST=\${TORCH_CUDA_ARCH_LIST:-9.0}
export MAX_JOBS=8
export UV_NO_SYNC=1
export UV_PROJECT_ENVIRONMENT=\$CONDA_PREFIX
export UV_PYTHON_PREFERENCE=only-system
export UV_CACHE_DIR=$UV_CACHE_DIR
export CARGO_HOME=$CARGO_HOME
export TORCH_EXTENSIONS_DIR=$CACHE_ROOT/torch_extensions
export HF_HOME=$CACHE_ROOT/huggingface
export ALPASIM_REPO=$ALPASIM_REPO
EOF

# 8. verify
"$ENV_MANAGER" deactivate && "$ENV_MANAGER" activate "$ALPASIM_ENV"
python - <<'PY'
import torch
print(f"torch {torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
import gsplat, alpasim_runtime, alpasim_mtgs, alpasim_wizard, utils_rs  # noqa: F401
print("gsplat", gsplat.__version__)
PY
echo ">> done. gsplat compiles its CUDA kernels on first use (~1 min, cached in $CACHE_ROOT/torch_extensions)."
echo ">> run 'bash setup/check_env.sh' on a GPU node to compile and test them."
