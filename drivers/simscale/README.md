# SimScale sample drivers

The `transfuser`, `diffusiondrive` and `gtrs_dense` drivers (`drivers/<name>.sh`) run these
packages in the alpasim env. They are copies of AlpaSim's
`e2e_challenge/sample_submission_simscale_navsim_<name>/navsim_<name>_challenge` at NVlabs/alpasim
`20235345d7ea9eed5d5156099cf219cd35b2e4dc` (branch `e2e_challenge`), so they don't change when
`setup/create_env.sh` clones a newer AlpaSim. Each `README.upstream.md` is that sample's README.
One change: on the PAI track, where the nuPlan cameras don't exist, each `driver.py` synthesizes
the CAM_L0/F0/R0 views it needs from the PAI cameras (`drivers/drivor/drivor_alpasim/pai_views.py`).
nuPlan sessions run the upstream code path unchanged.
They are Apache-2.0 (`LICENSE`, from AlpaSim).

- `gtrs_dense/assets/navsim_16384.npy` is the GTRS-Dense trajectory vocabulary (a Git LFS file
  upstream, SHA-256 `e8c29cfc…1261d`).
- The checkpoints aren't here. `setup/download_checkpoints.sh` puts them in `ckpts/simscale/`.

The upstream tests, Dockerfiles and container scripts were left out. The drivers' dependencies
(timm, opencv, dataclasses-json, alpasim_grpc) are in the alpasim env.
