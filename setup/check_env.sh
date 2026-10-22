#!/usr/bin/env bash
# Check the env on a GPU node: CUDA torch, and gsplat's CUDA kernels (compiled
# on first use, ~1 min, then cached).
#   bash setup/check_env.sh
set -euo pipefail
type module >/dev/null 2>&1 || [ "${ALPASIM_RELOGIN:-}" = "$0" ] || ALPASIM_RELOGIN=$0 exec bash -l "$0" "$@"
HPC_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$HPC_ROOT/lib/common.sh"
load_modules
activate_env

python - <<'PY'
import torch
print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, GPU available: {torch.cuda.is_available()}")
assert torch.cuda.is_available(), "torch sees no GPU (CPU-only wheel, or no GPU on this node?)"
print("GPU:", torch.cuda.get_device_name(0))

import gsplat
from gsplat import rasterization
n = 1000
means = torch.randn(n, 3, device="cuda") + torch.tensor([0.0, 0.0, 5.0], device="cuda")
quats = torch.nn.functional.normalize(torch.randn(n, 4, device="cuda"), dim=-1)
scales = torch.full((n, 3), 0.05, device="cuda")
opacities = torch.full((n,), 0.8, device="cuda")
colors = torch.rand(n, 3, device="cuda")
K = torch.tensor([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]], device="cuda")[None]
image, alpha, _ = rasterization(means, quats, scales, opacities, colors,
                                torch.eye(4, device="cuda")[None], K, 320, 240)
print(f"gsplat {gsplat.__version__}: rendered {tuple(image.shape)}, mean alpha {alpha.mean():.3f}")

import alpasim_runtime, alpasim_mtgs, alpasim_wizard, alpasim_controller, utils_rs  # noqa: F401
print("alpasim packages import fine")
PY
