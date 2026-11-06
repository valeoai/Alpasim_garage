# Alpasim_garage: AlpaSim E2E Challenge evaluation on HPC clusters, without Docker

Run the closed-loop evaluation of the
[AlpaSim E2E Challenge](https://github.com/NVlabs/alpasim/tree/e2e_challenge/e2e_challenge)
on a SLURM cluster that has **no Docker**. Both tracks are supported:

| Track | Renderer | Scenes | Runs on |
|---|---|---|---|
| **nuPlan (MTGS)** | Gaussian-splat MTGS, open source | NAVSIM `navtest`, 1,485 scenes of ~5.5 s | x86_64 and aarch64 (e.g. GH200) |
| **PAI (NuRec)** | NVIDIA NRE, closed-source image | NuRec `nurec_curated_val`, 441 scenes of ~20 s | x86_64 only (the NRE image is amd64-only) |

Upstream AlpaSim starts every service (renderer, controller, physics, runtime, driver) in its
own Docker container. This repo instead:

- builds **one micromamba (or conda) env** with the whole AlpaSim stack, with CUDA, GCC and
  Ninja taken from the cluster's environment modules;
- runs the AlpaSim wizard in config-only mode, then starts each service of the generated
  `docker-compose.yaml` as a **plain host process**. The PAI track's closed-source NRE
  renderer is the exception: it runs as its official image under **Apptainer/Singularity**;
- runs the **driver** (the policy under test) as a host process too. Five baselines are
  included: vanilla **DrivoR**, the three SimScale samples (**Latent TransFuser**,
  **DiffusionDrive**, **GTRS-Dense**), and AlpaSim's **starter** driver. You can add your own;
- shards a full evaluation **over all GPUs of several nodes** with a SLURM job array, then
  merges the shards and prints the challenge metrics.

On an x86_64 GPU, a run of this repo and a run of the official containers gave the **same
score and failure reason on every one of the 1,485 nuPlan scenes**
([Baseline results](#10-baseline-results)).

> **Scope.** This is for local development and scoring. A leaderboard submission is a Docker
> image pushed to the challenge's registry, which has to be built on a machine with Docker.

## Contents

0. [Quick start](#0-quick-start)
1. [How the evaluation works](#1-how-the-evaluation-works)
2. [Repository layout](#2-repository-layout)
3. [Requirements](#3-requirements)
4. [Configure for your cluster](#4-configure-for-your-cluster)
5. [Build the environment](#5-build-the-environment)
6. [Download the data](#6-download-the-data)
7. [Download the checkpoints](#7-download-the-checkpoints)
8. [Run a single scene (interactive)](#8-run-a-single-scene-interactive)
9. [Run a full evaluation on many GPUs (SLURM)](#9-run-a-full-evaluation-on-many-gpus-slurm)
10. [Baseline results](#10-baseline-results)
11. [Results, metrics and videos](#11-results-metrics-and-videos)
12. [Drivers, and adding your own](#12-drivers-and-adding-your-own)
13. [The PAI track: how NRE runs under Apptainer](#13-the-pai-track-how-nre-runs-under-apptainer)
14. [Troubleshooting](#14-troubleshooting)
15. [Known differences from the official evaluation](#15-known-differences-from-the-official-evaluation)
16. [Credits and license](#16-credits-and-license)

---

## 0. Quick start

```bash
git clone https://github.com/valeoai/Alpasim_garage.git && cd Alpasim_garage
$EDITOR config.sh                         # set PROJECT, ALPASIM_MODULES, SLURM_* (section 4)

# --- on a login node (internet) ---
bash setup/create_env.sh                  # env + AlpaSim checkout, ~15 min
bash setup/download_checkpoints.sh        # baseline checkpoints, ~3.6 GB
bash setup/download_nuplan_track.sh dev   # nuPlan data for the dev scene, ~36 GB (`full`: ~491 GB)

# --- on a GPU node (one GPU is enough) ---
bash setup/check_env.sh                   # CUDA torch + gsplat kernels
bash run_eval.sh starter dev              # pipeline check, no model
bash run_eval.sh drivor dev               # vanilla DrivoR on one scene, with video

# --- from a login node, once the full data is there: all 1,485 scenes on 2 nodes ---
bash slurm/submit.sh drivor 2 drivor_full
cat runs/drivor_full/metrics.txt          # written when the last shard finishes
```

For the PAI track, also see [section 6.2](#62-pai-track) (gated scenes, NRE image,
harmonizer, ego hood) and prefix the commands with `TRACK=pai`.

---

## 1. How the evaluation works

```
                    ┌────────────────────────────────────────────────┐
                    │ runtime (alpasim_runtime.simulate)             │
                    │ loads the scene, steps the rollout, calls      │
                    │ every service over gRPC, scores the rollout    │
                    └──┬──────────────────┬──────────────────┬───────┘
                       │                  │                  │
          ┌────────────▼─────┐  ┌─────────▼────────┐  ┌──────▼──────────────────┐
          │ renderer :6000   │  │ controller :6001 │  │ driver :6789            │
          │ nuPlan: MTGS     │  │ MPC: tracks the  │  │ YOUR POLICY (gRPC       │
          │ PAI: NRE (image) │  │ plan with a      │  │ egodriver service)      │
          │                  │  │ vehicle model    │  │ images+ego → trajectory │
          └──────────────────┘  └──────────────────┘  └─────────────────────────┘
          traffic: log replay (not reactive)       PAI only: + physics service
```

**nuPlan track.** Each scene is one NAVSIM `navtest` scene, about 5.5 s of a nuPlan log,
with an MTGS reconstruction of the surrounding area. For the first 0.5 s the ego follows
the recorded human path. Then, every 0.5 s (2 Hz), the renderer draws **8 cameras**
(`CAM_F0, L0, L1, L2, R0, R1, R2, B0`, 1920×1080) from the ego's simulated pose; the driver
receives the images, the ego pose, velocity and acceleration, and a route (map lanes about
40–80 m ahead), and returns a trajectory; the controller drives the ego along it for the
next 0.5 s. That makes **10 decisions, about 5 s** of closed-loop driving per scene.

**PAI track.** Same loop on NuRec reconstructions (`.usdz`) of NVIDIA's PhysicalAI-AV
recordings: longer scenes (~20 s), six 10 Hz f-theta cameras (`camera_front_wide_120fov`, …)
instead of the nuPlan rig, and a physics service. As in the recordings, the front camera shows the
hood of the ego car ([section 6.2](#62-pai-track)). The NAVSIM-trained baselines see nuPlan-like
views that their adapters synthesize from the PAI cameras.

**Scoring** (AlpaSim's `src/eval/src/eval/aggregation/scene_score.py`): a rollout scores
**0** on an at-fault collision (front or side), off-road, or leaving a **4 m** corridor around
the human path; otherwise `min(progress / 0.8, 1)`. The run score is the mean over scenes of
the mean over rollouts. The second main metric is **km per infraction**.
[Section 11](#11-results-metrics-and-videos) defines every reported metric.

---

## 2. Repository layout

```
config.sh                     site configuration: paths, modules, SLURM  ← edit this
run_eval.sh                   one driver, one GPU: the dev scene or a list of scenes
setup/
  create_env.sh               build the env and clone AlpaSim (login node, ~15 min)
  check_env.sh                test CUDA torch + gsplat kernels (GPU node)
  download_nuplan_track.sh    nuPlan-track data from Hugging Face (dev ~36 GB / full ~491 GB)
  download_checkpoints.sh     baseline checkpoints (DrivoR, SimScale) + NRE harmonizer
  download_pai_scenes.sh      PAI track: NuRec scenes from Hugging Face (gated) + ego hood
  pull_nre.sh                 PAI track: NRE renderer image → Apptainer .sif (x86_64)
drivers/
  README.md                   the driver contract; how to add your own
  starter.sh  drivor.sh  transfuser.sh  diffusiondrive.sh  gtrs_dense.sh
  drivor/drivor_alpasim/      DrivoR → AlpaSim gRPC adapter (+ PAI view synthesis, pai_views.py)
  simscale/                   the three SimScale sample drivers, from AlpaSim's e2e_challenge
slurm/
  submit.sh                   submit a sharded run as a job array
  job.sbatch                  one node: one shard per GPU
  run_shard.sh                one GPU: its slice of the scene list
  finalize.sh                 merge + score once every shard has finished
lib/
  common.sh                   module/env/driver helpers used by every script
  run_alpasim_local.py        runs the wizard's docker-compose.yaml as host processes
configs/
  nuplan_scenes/navtest_part001.yaml   100 nuPlan scenes of the first data tarball (quick check)
  pai_scenes/nurec_curated_val.yaml    the 441 PAI validation scenes
tools/
  wait_for_driver.py          block until a driver has loaded its model
  probe_driver.py             contract test of a live driver, no simulator needed
  summarize_run.py            challenge metrics (mean/var over rollouts); merges shards
  annotate_video.py           add a legend strip to a rollout video
  make_montage.py             join rollout videos of one drive in log order
  compare_videos.py           two drivers side by side on the same scenes
ckpts/                        checkpoints (git-ignored; filled by setup/download_checkpoints.sh)
runs/                         outputs (git-ignored)
```

---

## 3. Requirements

- A SLURM cluster with NVIDIA GPUs. Compute nodes **don't need internet**; the login node does
  (for setup and downloads).
- Environment modules (or the tools on `PATH`) for **CUDA 12.x**, **GCC ≥ 11**, **Ninja**, and,
  only while building the env, **Rust ≥ 1.74** (a module, or `rustup`).
- [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
  (or conda).
- Storage off `$HOME`: ~560 GB and ~115k files for the full nuPlan-track data, ~15 GB for the
  env and caches. The PAI track needs ~2 GB per scene (0.81 TB for the 441 validation scenes).
- **PAI track only:** x86_64 GPU nodes with an NVIDIA driver ≥ 570 (NRE is built against CUDA
  12.8), Apptainer or SingularityCE with `--nv`, ~12 GB for the NRE `.sif`, and a Hugging Face
  account that has accepted the terms of
  [`nvidia/PhysicalAI-Autonomous-Vehicles-NuRec`](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec).

---

## 4. Configure for your cluster

Every script sources `config.sh`. Edit the block at its top, or export any variable before
calling a script (exported variables also reach the SLURM jobs):

| Variable | Default | Meaning |
|---|---|---|
| `PROJECT` | **must be set** | base directory for code, data, envs and caches (project/scratch storage, not `$HOME`) |
| `ALPASIM_MODULES` | none | modules for CUDA 12.x, GCC and Ninja, e.g. `(GCC/13.3.0 Ninja/1.12.1 CUDA/12.9.0)` |
| `ALPASIM_BUILD_MODULES` | none | extra modules for building the env, e.g. `(Rust/1.78.0)` |
| `SLURM_ACCOUNT` | none | `--account` for the job array (omitted when empty) |
| `SLURM_PARTITION` | `gpu` | GPU partition |
| `GPUS_PER_NODE` | `4` | GPUs per node: one pipeline (shard) per GPU |
| `SLURM_RESOURCES` | `--ntasks-per-node=1 --gpus-per-node=$GPUS_PER_NODE --exclusive --time=03:00:00` | per-node request; one array task takes a whole node |

Paths derived from `PROJECT`, which you rarely need to change:

| Variable | Default | Meaning |
|---|---|---|
| `ALPASIM_REPO`, `ALPASIM_COMMIT` | `$PROJECT/code/alpasim`, `e121e59` | AlpaSim checkout (`e2e_challenge` branch), pinned to a commit with the [lateral-corridor fix](#corridor-fix), as the [baseline results](#10-baseline-results) are scored; `ALPASIM_COMMIT=2023534` reproduces the results scored before the fix, `ALPASIM_COMMIT=` keeps the branch head |
| `ALPASIM_NUPLAN_ROOT` | `$PROJECT/alpasim-nuplan-track` | nuPlan-track data |
| `ALPASIM_PAI_ROOT` | `$PROJECT/alpasim-pai` | PAI scenes, under `nre-artifacts/` |
| `NRE_IMAGE`, `NRE_SIF` | `nvcr.io/nvidia/nre/nre-ga:26.04`, `$PROJECT/images/nre-ga_26.04.sif` | PAI renderer image, and its Apptainer build |
| `CKPT_ROOT` | `ckpts` (in this repo) | driver checkpoints; `NRE_HARMONIZER_DIR` = `$CKPT_ROOT/nvidia_renderer_ckpt` |
| `CACHE_ROOT` | `$PROJECT/.cache` | pip/uv/cargo/Hugging Face/torch-extension caches |
| `ENV_MANAGER`, `ALPASIM_ENV` | `micromamba`, `alpasim` | `conda` also works (then set `CONDA_ROOT`); `ALPASIM_ENV` may be a path (prefix env) |
| `MAMBA_EXE`, `MAMBA_ROOT_PREFIX` | `micromamba` on `PATH`, `$PROJECT/env/micromamba` | micromamba binary and root |
| `DRIVOR_CHECKPOINT_PATH` | `$CKPT_ROOT/drivor_vanilla/drivor_Nav1_25epochs.pth` | DrivoR checkpoint |

Without a module system (e.g. a cloud VM), the scripts skip the modules with a warning, so CUDA,
GCC, Ninja and (for `create_env.sh`) Rust must already be on `PATH`. Without SLURM, run
`run_eval.sh` once per GPU with its own `GPU`/`BASEPORT`/`DRIVER_PORT` (section 8).

---

## 5. Build the environment

On a **login node**:

```bash
bash setup/create_env.sh
```

It mirrors AlpaSim's `Dockerfile`, in an env instead of an image:

1. creates the env with `python=3.12 uv ffmpeg` from conda-forge;
2. clones AlpaSim (`e2e_challenge`, at `ALPASIM_COMMIT`) to `$ALPASIM_REPO`;
3. compiles the gRPC protos and runs `uv sync` into the env with the `all` and `mtgs` extras
   (skipping PyQt5, which has no aarch64 wheel and is only used by optional visualization tools);
4. reinstalls torch 2.8.0 / torchvision 0.23.0 from PyTorch's `cu129` index (AlpaSim's lockfile
   pulls the PyPI wheel, which is CPU-only on aarch64);
5. installs `timm==1.0.15` for DrivoR;
6. writes an activation hook that sets `UV_NO_SYNC=1` (so a stray `uv run` can't put CPU torch
   back), `TORCH_CUDA_ARCH_LIST`, `TORCH_EXTENSIONS_DIR` and `ALPASIM_REPO`.

Then, on a **GPU node**, compile and test gsplat's CUDA kernels (about 1 min the first time,
then cached in `$CACHE_ROOT/torch_extensions`):

```bash
srun -p <gpu-partition> --gres=gpu:1 -t 00:20:00 --pty bash -l
bash setup/check_env.sh
# torch 2.8.0+cu129, CUDA 12.9, GPU available: True
# gsplat 1.5.3: rendered (1, 240, 320, 3), mean alpha 0.412
```

To use the env interactively: `micromamba activate alpasim` (with your `MAMBA_ROOT_PREFIX`).

---

## 6. Download the data

Run downloads on a **login node**, inside `tmux` or with `nohup`. All of them can be resumed:
rerun the same command after an interruption.

### 6.1 nuPlan track

Source: the public Hugging Face dataset
[`OpenDriveLab/AlpasimChallenge2026_nuplan_track`](https://huggingface.co/datasets/OpenDriveLab/AlpasimChallenge2026_nuplan_track)
(no token needed).

```bash
bash setup/download_nuplan_track.sh dev        # dev scene + the 100 scenes of part001: 3 files, ~36 GB
nohup bash setup/download_nuplan_track.sh full > download.log 2>&1 &   # all of navtest: 17 files, ~491 GB
```

The script handles one tarball at a time (download → extract → delete), so it needs at most
one shard (~35 GB) of extra disk. Finished files are recorded in `$ALPASIM_NUPLAN_ROOT/.done/`.
The full set takes about 1.5–2 h; at the end it should report 1,491 scenes with assets, 1,491
configs and 12,148 trajdata scenes. Every scene of `navtest_full.yaml` (1,485) is covered.

| Folder | Contents |
|---|---|
| `nuplan_test/` | trajdata scene cache: ego and agent tracks, timing, calibration (~97k files) |
| `navtest/configs/` | one YAML per scene: log, frames, city, reconstructed area |
| `navtest/assets/` | the MTGS reconstructions: one ~420 MB `background/*.ckpt` per scene, plus a road height map |

No raw nuPlan images, lidar or maps are needed: the renderer produces the images, and the
off-road check uses road polygons shipped with AlpaSim.

### 6.2 PAI track

**1. Hugging Face access.** Accept the terms on the
[NuRec dataset page](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec),
then log in on the login node (`hf auth login` or `huggingface-cli login`, or put a read token in
`~/.cache/huggingface/token`, or export `HF_TOKEN`).

**2. Scenes** (`.usdz`, ~2 GB each, into `$ALPASIM_PAI_ROOT/nre-artifacts/all-usdzs`).
`download_pai_scenes.sh` runs the wizard in config-only mode, which downloads whatever is
missing. It takes the same scene overrides as `run_eval.sh`:

```bash
bash setup/download_pai_scenes.sh                              # the dev scene (smoke test), ~2 GB
bash setup/download_pai_scenes.sh +nurec_scenes=curated_val    # the 441 validation scenes, 0.81 TB
bash setup/download_pai_scenes.sh "scenes.scene_ids=['clipgt-...','clipgt-...']"
```

**3. NRE renderer image** (public on NGC, no key needed; ~12 GB; building it needs about twice
that in `$APPTAINER_CACHEDIR`):

```bash
bash setup/pull_nre.sh          # docker://nvcr.io/nvidia/nre/nre-ga:26.04 -> $NRE_SIF
```

If your cluster can't build the image on one node (e.g. the internet-connected nodes kill
`mksquashfs` for its memory use), pull it there anyway: the layers land in the cache before
the squash fails. Then build offline on a node with enough memory:
`NRE_OCI_LAYOUT=<cache>/blob bash setup/pull_nre.sh`. Some sites also require registering
images in an approved directory: point `NRE_SIF` there.

**4. Harmonizer checkpoint.** The official config runs NRE with `--enable-harmonizer`, a
post-processing model NRE downloads on first use, which fails on offline compute nodes.
`setup/download_checkpoints.sh` fetches it (section 7), and the renderer gets it read-only at
NRE's default cache path, so its command stays the official one.

**5. Ego hood.** As in the challenge, the renderer draws the hood of the recording car (rig
`hyperion_8_1`) over every `camera_front_wide_120fov` frame sent to the driver; the other
cameras, and the nuPlan track, never get one. The hood images are not in the Hugging Face
dataset: they ship in the AlpaSim repo (`data/nre-artifacts/ego-hoods/`), and the renderer reads
them from `$ALPASIM_PAI_ROOT/nre-artifacts/ego-hoods`. `download_pai_scenes.sh` (step 2) copies
them there from the AlpaSim checkout. If they are missing, NRE silently renders no hood, so
`run_eval.sh` and `slurm/submit.sh` stop with an error instead. `EGO_HOOD=0` runs without the hood
(`runtime.simulation_config.ego_mask_rig_config_id=null`), as the PAI results of
[section 10](#10-baseline-results) were scored.

---

## 7. Download the checkpoints

```bash
bash setup/download_checkpoints.sh                      # all of the below
bash setup/download_checkpoints.sh drivor transfuser    # or just some
```

| Name | Checkpoint (under `$CKPT_ROOT`) | Source |
|---|---|---|
| `drivor` | `drivor_vanilla/drivor_Nav1_25epochs.pth` (292 MB, SHA-256 checked) | [valeoai/DrivoR release](https://github.com/valeoai/DrivoR/releases/download/model_weights/drivor_Nav1_25epochs.pth) |
| `transfuser` | `simscale/SimScale_ckpts/LTF/ltf_sim_navtest.ckpt` | [OpenDriveLab/SimScale](https://huggingface.co/datasets/OpenDriveLab/SimScale) |
| `diffusiondrive` | `simscale/SimScale_ckpts/DiffusionDrive/diffusiondrive_sim_navhard.ckpt` | OpenDriveLab/SimScale |
| `gtrs_dense` | `simscale/SimScale_ckpts/GTRS_Dense/gtrs_dense_resnet_sim_reward_navhard.ckpt` (vocabulary in `drivers/simscale/gtrs_dense/assets/`) | OpenDriveLab/SimScale |
| `harmonizer` | `nvidia_renderer_ckpt/harmonizer_nontemporal.pt` (1.4 GB, PAI track only) | [nvidia/DiffusionHarmonizer](https://huggingface.co/nvidia/DiffusionHarmonizer) |

The `starter` driver needs none. Each driver script also takes its own path variable
(`DRIVOR_CHECKPOINT_PATH`, `LTF_CHECKPOINT_PATH`, `DIFFUSIONDRIVE_CHECKPOINT_PATH`,
`GTRS_CHECKPOINT_PATH`) to evaluate your own checkpoint of the same architecture.

---

## 8. Run a single scene (interactive)

On a **GPU node** (one GPU is enough):

```bash
bash run_eval.sh starter dev                 # nuPlan pipeline check, no model (a few min)
bash run_eval.sh drivor dev                  # one scene with video (~2–6 min)
TRACK=pai bash run_eval.sh starter dev       # PAI pipeline check (~6 min)
TRACK=pai bash run_eval.sh drivor dev        # PAI dev scene (~8 min)
```

`run_eval.sh DRIVER [PRESET] [RUN] [wizard overrides…]`:

1. starts the driver in the background (`drivers/<DRIVER>.sh`, log in `runs/<RUN>/driver.log`);
2. runs the AlpaSim wizard with `wizard.run_method=NONE`, which writes the service configs and
   `docker-compose.yaml` but starts nothing;
3. waits until the driver has loaded its model (`tools/wait_for_driver.py`);
4. starts the renderer and controller (and, on PAI, physics) from the compose file, waits for
   their ports, then runs the runtime in the foreground; everything is stopped when it exits;
5. prints and saves the metrics (`runs/<RUN>/metrics.txt`, `metrics.json`).

| Argument | Values |
|---|---|
| `DRIVER` | `starter`, `drivor`, `transfuser`, `diffusiondrive`, `gtrs_dense`, or the path to your own driver script |
| `PRESET` | nuPlan: `dev` (1 scene, with video) or `full` (1,485 scenes on one GPU: hours; use section 9). PAI: `dev` (1 scene; other scenes via `scenes.scene_ids=[...]`) |
| `RUN` | output folder under `runs/` (default `<driver>_<preset>_<date>`) |
| overrides | any AlpaSim wizard (Hydra) setting |

```bash
# chosen scenes (nuPlan IDs: $ALPASIM_REPO/src/wizard/configs/nuplan_scenes/*.yaml)
bash run_eval.sh drivor dev two_scenes "scenes.scene_ids=['<scene_id>','<scene_id>']"
TRACK=pai bash run_eval.sh drivor dev pai_one "scenes.scene_ids=['clipgt-...']"
# several pipelines on one node: give each its own GPU and ports
GPU=1 BASEPORT=6100 DRIVER_PORT=7001 bash run_eval.sh diffusiondrive dev
```

| Environment variable | Default | Effect |
|---|---|---|
| `TRACK` | `nuplan` | `nuplan` or `pai` |
| `EGO_HOOD` | `1` | PAI: draw the ego hood on the front camera, as in the challenge; `0` turns it off (section 6.2) |
| `GPU` | `0` | GPU for the renderer and the driver |
| `BASEPORT` | `6000` | renderer port; controller on `BASEPORT+1` (physics `+2`) |
| `DRIVER_PORT` | `6789` | driver gRPC port |
| `N_ROLLOUTS` | `1` | rollouts per scene |
| `VIDEO` | dev: on, otherwise off | `eval.video.render_video` |
| `WIZARD_OVERRIDES` | – | extra wizard overrides, space-separated |
| `RUNS_ROOT` | `./runs` | where run folders go |

**Timing.** The MTGS renderer starts in about 10 s with a warm file cache, but can take ~5 min
on a cold filesystem while it indexes the 12k-scene trajdata cache. A nuPlan rollout then takes
8–13 s (20–30 s with video). NRE starts in 30–40 s; a PAI scene takes 3–5 min.

---

## 9. Run a full evaluation on many GPUs (SLURM)

`slurm/submit.sh DRIVER NODES [RUN] [sbatch args…]` submits a job array. Each array task takes
**one node** and runs **one independent pipeline per GPU** (driver + renderer + controller +
runtime). The scene list is sorted and split round-robin into `NODES × GPUS_PER_NODE` shards.
When the last shard finishes, the merged summary is written to `runs/<RUN>/metrics.txt`.

Submit from a **login node**.

**nuPlan track, full navtest (1,485 scenes):**

```bash
SHARD_LIMIT=2 bash slurm/submit.sh drivor 2 smoke --time=00:30:00   # smoke test: 2 scenes per shard
bash slurm/submit.sh drivor 2 drivor_full                           # e.g. 2 nodes × 4 GPUs: ~35–65 min
N_ROLLOUTS=3 bash slurm/submit.sh diffusiondrive 4 dd_full_x3       # stochastic driver: 3 rollouts per scene

# the 100 scenes of the `dev` download, as a quick check before the full download
SCENE_LIST=$PWD/configs/nuplan_scenes/navtest_part001.yaml bash slurm/submit.sh transfuser 1 ltf_part001
```

**PAI track, validation (441 scenes):**

```bash
TRACK=pai SCENE_LIST=$PWD/configs/pai_scenes/nurec_curated_val.yaml \
  bash slurm/submit.sh drivor 8 pai_val_drivor --time=02:00:00     # 8 nodes × 4 GPUs: ~70 min
```

A PAI scene takes 3–5 min of one GPU, so the validation set is about 20–40 GPU-hours.

| Variable | Effect |
|---|---|
| `SCENE_LIST=file.yaml` | take the scenes from that file (`scenes.scene_ids`) instead of `navtest_full` (use an absolute path) |
| `TRACK=pai` | run the PAI track (needs a `SCENE_LIST`) |
| `SHARD_LIMIT=N` | keep only the first N scenes of each shard (smoke tests) |
| `N_ROLLOUTS`, `VIDEO`, `EGO_HOOD`, `WIZARD_OVERRIDES`, `DRIVOR_CHECKPOINT_PATH`, … | passed on to every shard |

- **Monitoring:** `squeue -u $USER`; per node `runs/<RUN>/slurm-<job>_<task>.out`; per shard
  `runs/<RUN>/shard_XXX.out` and `runs/<RUN>/shard_XXX/{driver.log,wizard.log,txt-logs/}`.
- **Resuming:** resubmit the same command with the same `RUN` and the same `NODES`. Shards that
  already have results are skipped; unfinished ones restart. `submit.sh` refuses a different
  node count for an existing `RUN`, because that would change the split. To rerun only some
  tasks, add `--array=<indices>` (and e.g. `--exclude=<node>` for a bad node).
- **Time limit:** 3 h per node by default (`SLURM_RESOURCES`); override it after the RUN name,
  e.g. `--time=01:00:00`. Any other sbatch option (`--qos`, `--constraint`, …) goes there too.
- **By hand:** `python tools/summarize_run.py runs/<RUN> --scenes navtest_full` (or
  `--scenes configs/pai_scenes/nurec_curated_val.yaml`), in the `alpasim` env.

---

## 10. Baseline results

> [!WARNING]
> **The PAI-track results in this section were scored without the ego hood.** Back then the
> renderer's hood folder was empty, so the front camera showed no hood, unlike in the challenge.
> PAI runs now draw the hood by default ([section 6.2](#62-pai-track)), so their scores are not
> directly comparable to these tables; set `EGO_HOOD=0` to reproduce them. The nuPlan results are
> not affected: the nuPlan track has no hood.

All scored **with the lateral-corridor fix** (AlpaSim `1b024f0`, see the note below), which the
`config.sh` default `e121e59` includes, and the checkpoints of section 7. Scene score is the
challenge's main metric; see section 11 for the others. The simulation is deterministic, so a
deterministic driver gives identical rollouts; only DiffusionDrive (which samples its diffusion
noise) varies between rollouts. The results scored before the fix are kept at the end of this
section.

The runs below are on x86_64 A100 nodes (IT4I Karolina). They were driven at `2023534` and scored
with the `ground_truth.py` of `1b024f0`; `2023534`, `1b024f0` and `e121e59` differ only in that
scorer and in the challenge CLI, so a run at the `e121e59` default gives the same scores.

### nuPlan track: full navtest (1,485 scenes)

1 rollout per scene, except DiffusionDrive (3):

| Driver | Scene score (%) ↑ | km / infr. ↑ | Coll. at-fault (%) ↓ | Off-road (%) ↓ | Corridor (%) ↓ | Progress (%) ↑ | Pass (%) ↑ | Rear-ended (%) |
|---|---|---|---|---|---|---|---|---|
| DrivoR (vanilla) | **88.74** | **0.528** | 1.69 | **3.80** | **1.90** | 89.55 | **91.99** | 3.53 |
| Latent TransFuser (SimScale) | 82.81 | 0.296 | 1.36 | 8.27 | 4.75 | 88.12 | 85.59 | 1.56 |
| GTRS-Dense (SimScale) | 82.55 | 0.452 | 3.05 | 4.07 | 9.63 | **94.11** | 82.90 | 1.49 |
| DiffusionDrive (SimScale) | 80.02 ± 0.20 | 0.239 | **1.31** | 10.40 | 5.20 | 87.23 | 82.96 | 1.72 |
| Starter (straight line) | 53.97 | 0.116 | 5.02 | 16.54 | 13.08 | 70.41 | 67.54 | 6.10 |

± is the standard deviation over the 3 rollout repetitions (0 for the other drivers). Scores
include 10 scenes that crash for every driver (section 11), about 0.7 points. Expect a few tenths
of a point between GPU types (renderings differ numerically and move a few scenes); GTRS-Dense
is the most sensitive, with 82.55 here against 82.97 on GH200 nodes even before the fix.

**Check against the official container evaluation.** Vanilla DrivoR, 1 rollout per scene, on
x86_64 GPUs, scored before the fix (the fix changes the scorer only, so the check still holds):

| Setup | Scene score (%) ↑, before fix | km / infr. ↑ | Coll. at-fault (%) ↓ | Off-road (%) ↓ | Corridor (%) ↓ | Progress (%) ↑ | Failures: at-fault / off-road / corridor / route error |
|---|---|---|---|---|---|---|---|
| Official containers, RTX PRO 6000 (Blackwell) | 88.35 | 0.549 | 1.63 | 3.66 | 2.58 | 89.57 | 24 / 54 / 37 / 10 |
| This repo, same GPU | 88.35 | 0.549 | 1.63 | 3.66 | 2.58 | 89.57 | 24 / 54 / 37 / 10 |
| This repo, H100 | 88.13 | 0.529 | 1.76 | 3.73 | 2.58 | 89.61 | 26 / 55 / 37 / 10 |

On the same GPU, every one of the 1,485 scenes gets the same score and failure reason in both
setups, and AlpaSim's own `local_evaluation/evaluate.py --track nuplan --without-references`
gives 0.88354 for both. Across GPU types, renderings differ numerically, which moves a few
scenes (here 3 of 1,485); on one GPU type, a rerun reproduces every scene exactly.

> **Comparing with other numbers:** check their AlpaSim commit. Commit `0bb4c4b` (#181,
> 2026-09-14) turned on the off-road failure for the nuPlan track; scoring the run above without
> it gives 91.01 instead of 88.35. Commit `1b024f0` (#215, 2026-10-20) is the corridor fix below.

<a id="corridor-fix"></a>
> **Lateral-corridor fix (`1b024f0`, #215, 2026-10-20).** A rollout fails when the ego ends
> ≥ 4 m *sideways* from the human's path; driving further *along* it is allowed. Past the end of
> the path, AlpaSim up to `2023534` measured the sideways part against the heading of the last
> recorded segment, which can point anywhere when the recording ends with near-duplicate poses,
> and position jitter could also make the plain distance (overshoot included) count as sideways.
> An ego that reaches the end of the human path could therefore fail the corridor while
> driving in its lane. The fix takes the heading over the last metre of motion and measures the
> distance to the path extended by a ray from its endpoint. It only removes such false corridor
> exits, so scores can only go up, and drivers that keep pace with the human gain the most: on
> the PAI track, vanilla DrivoR goes from 29.16 to 36.22 and GTRS-Dense from 19.22 to 26.63. It
> changes `src/eval/src/eval/scorers/ground_truth.py` only; the next commit, `e121e59`, adds
> report uploading to the challenge CLI and changes no score. For an existing setup,
> `git -C $ALPASIM_REPO fetch && git -C $ALPASIM_REPO checkout e121e59` is enough: AlpaSim is
> installed in editable mode, so no rebuild is needed.

### nuPlan track: 100-scene quick check (`navtest_part001`)

The 100 scenes of the first asset tarball (`configs/nuplan_scenes/navtest_part001.yaml`), which
the `dev` download already contains. One H100 node (4 shards) runs them in 6–7 min per driver.
Use it to check a new site; 100 scenes are too few to rank the learned drivers. The numbers are
these 100 scenes of the full runs above (A100, 1 rollout per scene, DiffusionDrive 3):

| Driver | Scene score (%) ↑ | km / infr. ↑ | Coll. at-fault (%) ↓ | Off-road (%) ↓ | Corridor (%) ↓ | Progress (%) ↑ | Pass (%) ↑ |
|---|---|---|---|---|---|---|---|
| Latent TransFuser (SimScale) | 90.67 | 0.911 | 0.00 | 3.00 | 4.00 | 87.14 | 93.00 |
| DrivoR (vanilla) | 88.51 | 0.907 | 2.00 | 1.00 | 5.00 | 88.57 | 92.00 |
| DiffusionDrive (SimScale) | 86.49 | 0.572 | 0.00 | 4.67 | 6.00 | 85.42 | 89.33 |
| GTRS-Dense (SimScale) | 82.27 | 0.502 | 4.00 | 2.00 | 11.00 | 92.28 | 83.00 |
| Starter (straight line) | 60.83 | 0.269 | 3.00 | 7.00 | 15.00 | 70.41 | 77.00 |

### PAI track: validation (`nurec_curated_val`, 441 scenes)

**Without the ego hood** (`EGO_HOOD=0`; see the warning at the top of this section). 1 rollout
per scene, except DiffusionDrive (3, ± the standard deviation over them):

| Driver | Scene score (%) ↑ | km / infr. ↑ | Coll. at-fault (%) ↓ | Off-road (%) ↓ | Corridor (%) ↓ | Progress (%) ↑ | Pass (%) ↑ | Rear-ended (%) |
|---|---|---|---|---|---|---|---|---|
| DrivoR (vanilla) | **36.22** | 0.212 | 15.65 | **18.14** | 21.77 | 47.46 | **49.43** | 18.59 |
| Latent TransFuser (SimScale) | 33.16 | 0.227 | **10.43** | 21.09 | 26.53 | 47.42 | 47.17 | 21.09 |
| DiffusionDrive (SimScale) | 29.55 ± 0.32 | 0.178 | 10.81 | 23.89 | **21.16** | 41.69 | 49.13 | 23.05 |
| GTRS-Dense (SimScale) | 26.63 | 0.174 | 13.83 | 27.89 | 35.83 | 48.73 | 28.57 | 3.85 |
| Starter (straight line) | 22.10 | **0.270** | 18.37 | 22.68 | 22.45 | **51.69** | 36.73 | 14.74 |

The NAVSIM baselines were trained on nuPlan cameras and see nuPlan views synthesized from the
PAI rig, hence the low scores. Unlike on nuPlan, camera drivers vary by about ±0.5 points
between full PAI runs, even on one GPU type, so compare means over several runs. The PAI
scenes are not comparable to the nuPlan ones (about 20 s instead of 5 s).

The `dev` scene gives a quick pipeline check: `starter` scores 39.63 (progress 31.70,
rear-ended 100%, `failures: none=1`) and vanilla DrivoR 100.00, the same before and after the
fix (neither leaves the corridor). DrivoR also scores 100.00 with the ego hood (its rollout
changes slightly); `starter` ignores the cameras, so the hood can't change its score.

<details>
<summary><b>Results scored before the corridor fix</b> (AlpaSim <code>2023534</code>; set <code>ALPASIM_COMMIT=2023534</code> to reproduce them)</summary>

**nuPlan track, full navtest (1,485 scenes).** aarch64 GH200 nodes; 3 rollouts per scene,
except vanilla DrivoR (1, deterministic):

| Driver | Scene score (%) ↑, before fix | km / infr. ↑ | Coll. at-fault (%) ↓ | Off-road (%) ↓ | Corridor (%) ↓ | Progress (%) ↑ | Pass (%) ↑ | Rear-ended (%) |
|---|---|---|---|---|---|---|---|---|
| DrivoR (vanilla) | **87.86** | **0.516** | 1.83 | **3.80** | **2.71** | 89.58 | **91.11** | 3.53 |
| GTRS-Dense (SimScale) | 82.97 | 0.484 | 2.58 | 4.07 | 9.69 | **94.03** | 83.30 | 1.63 |
| Latent TransFuser (SimScale) | 82.46 | 0.296 | 1.36 | 8.27 | 5.15 | 88.11 | 85.25 | 1.63 |
| DiffusionDrive (SimScale) | 79.66 ± 0.28 | 0.240 | **1.24** | 10.40 | 5.60 | 87.22 | 82.56 | 1.60 |
| Starter (straight line) | 53.57 | 0.116 | 5.02 | 16.54 | 13.49 | 70.41 | 67.14 | 6.10 |

**nuPlan track, 100-scene quick check (`navtest_part001`).** One H100 node (4 shards), 1 rollout
per scene:

| Driver | Scene score (%) ↑, before fix | km / infr. ↑ | Coll. at-fault (%) ↓ | Off-road (%) ↓ | Corridor (%) ↓ | Progress (%) ↑ | Pass (%) ↑ |
|---|---|---|---|---|---|---|---|
| Latent TransFuser (SimScale) | 90.68 | 0.913 | 0.00 | 3.00 | 4.00 | 87.19 | 93.00 |
| DrivoR (vanilla) | 86.40 | 0.681 | 2.00 | 2.00 | 6.00 | 88.71 | 90.00 |
| DiffusionDrive (SimScale) | 85.40 | 0.445 | 1.00 | 5.00 | 6.00 | 85.56 | 88.00 |
| GTRS-Dense (SimScale) | 81.27 | 0.503 | 4.00 | 2.00 | 12.00 | 92.47 | 82.00 |
| Starter (straight line) | 59.83 | 0.269 | 3.00 | 7.00 | 16.00 | 70.41 | 76.00 |

**PAI track, validation (`nurec_curated_val`, 441 scenes).** x86_64 A100 nodes, 1 rollout per
scene; mean ± half the gap between two full runs (on an H100 node, `starter` reproduces its row
exactly and vanilla DrivoR scores 30.06):

| Driver | Scene score (%) ↑, before fix | km / infr. ↑ | Coll. at-fault (%) ↓ | Off-road (%) ↓ | Corridor (%) ↓ | Progress (%) ↑ | Pass (%) ↑ | Rear-ended (%) |
|---|---|---|---|---|---|---|---|---|
| DrivoR (vanilla) | **29.16** ± 0.29 | 0.192 | 16.10 | **19.62** | 27.55 | 46.17 | 42.29 | 18.03 |
| Latent TransFuser (SimScale) | 27.65 ± 0.03 | 0.225 | 10.88 | 20.98 | 32.09 | 47.38 | 41.72 | 20.98 |
| DiffusionDrive (SimScale) | 26.46 ± 0.42 | 0.180 | **10.21** | 24.04 | **24.15** | 41.56 | **46.49** | 23.92 |
| GTRS-Dense (SimScale) | 19.22 ± 0.02 | 0.172 | 14.63 | 27.21 | 43.43 | 48.57 | 21.21 | 3.74 |
| Starter (straight line) | 16.65 ± 0.00 | **0.270** | 18.37 | 22.68 | 28.34 | **51.69** | 31.29 | 14.74 |

</details>

---

## 11. Results, metrics and videos

Each run folder, or each `shard_XXX/` of a sharded run, contains:

```
aggregate/results-summary.json     one entry per rollout: score, status, failure_reason, all metrics
aggregate/videos/                  rollout videos (when rendered)
rollouts/<scene>/<rollout>/        rollout.asl (full log), metrics.parquet (per timestep)
driver.log  wizard.log  txt-logs/  logs of every service
docker-compose.yaml, generated-*.yaml   the configs the wizard generated
```

and the run folder gets `metrics.txt` (the printed table) and `metrics.json`:

```
Metric                    value
───────────────────────────────
Scene score (%) ↑         88.74
km / infr. ↑              0.528
Coll. at-fault (%) ↓       1.69
...
failures:  none=1366, offroad=56, left_corridor_laterally=28, collision_at_fault=25, ...
driver:    cached_plan=0, drivor_inference=14750, dynamic_state_fallback=0, inference_error=0, straight_fallback=0
```

With `N_ROLLOUTS>1`, each metric is also computed per rollout repetition (the k-th rollout of
every scene, scored as a separate run), and the table shows the mean, sample variance and
standard deviation over repetitions, plus each repetition's value.

### The metrics

The first two are the **main challenge metrics** (AlpaSim's
`src/eval/src/eval/aggregation/scene_score.py`; OPTED,
[arXiv:2609.20756](https://arxiv.org/abs/2609.20756), Sec. IV-b). Every metric is aggregated
along a rollout only **until its first incident** (collision or off-road); after that, progress
and distance are frozen.

| Metric | Definition | How to read it |
|---|---|---|
| **Scene score (%)** ↑ | Per rollout: **0** if it fails (at-fault collision, off-road or corridor exit), otherwise `min(progress / 0.8, 1)`; a pass scores 1 when the human drove < 5 m. Averaged over a scene's rollouts, then over scenes, ×100. | The main metric. It rewards not failing and not being too slow; above 80% progress there is no extra credit. No comfort, TTC or traffic-rule terms. |
| **km / infr.** ↑ | Total km driven (until each rollout's first incident) ÷ number of infractions (at-fault collision or off-road, at most one per rollout). | The safety metric (the leaderboard's "At-Fault Distance"). Corridor exits don't count. |
| **Coll. at-fault (%)** ↓ | % of rollouts where the ego hits another vehicle **with its front or side**. | Rear hits don't count: traffic replays its log and can't react. |
| **Off-road (%)** ↓ | % of rollouts where the ego's footprint leaves the drivable area. | Usually cutting a corner, or drifting onto a curb or parking area. |
| **Corridor (%)** ↓ | % of rollouts ending ≥ **4 m sideways** from the human's logged path. | Not a safety rule: it keeps the ego where the reconstruction is reliable. It can penalize legitimate choices, such as another lane through a turn. |
| **Progress (%)** ↑ | Distance along the human path relative to what the human covered in the same time, clipped to [0, 100]. | Assertiveness; it only raises the score up to 80%. |
| **Pass (%)** ↑ | % of rollouts with no failure. | Scene score without the progress term. |
| **Rear-ended (%)** | % of rollouts hit from behind by replayed traffic. | Not a failure, but it ends the rollout early; high values mean the driver brakes where the human didn't. |

- **Crashed rollouts.** A rollout that crashes before it is scored counts as a failure with
  score 0, as in AlpaSim's own aggregation, and is left out of the other columns. On the nuPlan
  track, 10 of the 1,485 scenes crash for every driver with `Waypoint N fails sanity check:
  route folds back on itself` (AlpaSim's route generator; all are near-stationary scenes).
- **Driver counters.** The `driver:` line sums the per-session counters that the DrivoR and
  SimScale adapters log. `straight_fallback` and `inference_error` should be **0**; otherwise
  some decisions didn't come from the model.
- **Leaderboard ranking.** The official leaderboard fits Drive-IRT (an item-response model) over
  all submissions' per-scene scores on a **private** scene set and reports a Policy Capability
  Score ([documentation](https://ln2697.github.io/navhard-leaderboard-2/docs/)).
  `$ALPASIM_REPO/e2e_challenge/local_evaluation/evaluate.py` runs the same fit locally (it needs
  at least 6 runs).

### Videos

The `dev` preset renders a video; for other runs add `VIDEO=1`. The layout shows a bird's-eye
map (top left), live metrics (top right) and the front camera (bottom).

```bash
python tools/annotate_video.py runs/<RUN>/aggregate/videos/all/<file>.mp4 --slow 2   # legend strip + slowed copy
python tools/make_montage.py runs/<RUN> --slow 2 --name my_montage                   # consecutive scenes of a drive
python tools/compare_videos.py runs/<RUN_A> runs/<RUN_B> --labels A B                # two drivers side by side
```

Map colours: **orange** the driver's plan; **solid green box** the ego (the map is ego-centred,
ego-up); **faint green box and line** the human car and its logged path; **green dashed** the
start of the route given to the driver; **grey boxes** other vehicles; **blue** lanes.

---

## 12. Drivers, and adding your own

| Driver | Model | Cameras used | Startup | Notes |
|---|---|---|---|---|
| `starter` | none: straight line at the current speed | – | instant | pipeline test only |
| `drivor` | vanilla [DrivoR](https://github.com/valeoai/DrivoR) | F0, B0, L0, R0 | ~40–80 s | adapter in `drivers/drivor/` |
| `transfuser` | SimScale Latent TransFuser | L0, F0, R0 | ~30 s | AlpaSim sample, `drivers/simscale/` |
| `diffusiondrive` | SimScale DiffusionDrive | L0, F0, R0 | ~30 s | AlpaSim sample; stochastic |
| `gtrs_dense` | SimScale GTRS-Dense (ResNet, reward) | L0, F0, R0 | ~30 s | AlpaSim sample |

On the PAI track, the four learned drivers see nuPlan pinhole views synthesized from the PAI
f-theta cameras (`drivers/drivor/drivor_alpasim/pai_views.py`): each view's pixel rays are
rotated into the rig frame and sampled from the PAI camera that sees them closest to its optical
axis; `CAM_B0` is stitched from the two rear cameras.

A driver is any script that runs a gRPC server implementing AlpaSim's
`egodriver.EgodriverService` in the foreground. The easiest route for a NAVSIM model is to copy
`drivers/drivor/` and replace `policy.py` and `preprocessing.py`. See
[`drivers/README.md`](drivers/README.md) for the full contract, then test it without the
simulator and run it:

```bash
bash drivers/mymodel.sh &
python tools/wait_for_driver.py localhost:6789 && python tools/probe_driver.py --address localhost:6789
kill %1
bash run_eval.sh path/to/mymodel.sh dev
```

---

## 13. The PAI track: how NRE runs under Apptainer

NRE ships only as an amd64 image whose code is deliberately obfuscated, so it runs **as the
whole official image**, never unpacked. `lib/run_alpasim_local.py` starts every compose service
that has no `build` key (only the renderer) with `apptainer exec --nv --cleanenv` (or
`singularity`, if that is what the cluster has). Each compose volume is bound at its container
path, the image's working directory is used, and the renderer command is the wizard's,
unchanged. Every other service is a host process. Three things Docker doesn't need:

- **`PYTHONDONTWRITEBYTECODE=1` in the renderer.** Python writes `.pyc` files into the image
  tree on import; Apptainer's `--writable-tmpfs` is capped (often 64 MB), and when it fills up,
  NRE's loader reports the `No space left on device` as `No module named 'nre.nrm'`.
- **No host preloads in the container.** Variables such as `SINGULARITYENV_LD_PRELOAD` (set by
  job trackers like XALT) and host bind paths are removed from the renderer's environment.
- **Host paths in generated configs.** The host services' configs refer to container mount
  points (e.g. the runtime's `usdz.data_dir=/mnt/nre-data/...`); they are rewritten to the
  host paths of the compose volumes, as the commands already are.

The renderer's `--egocar-hood-dir` is bound from `$ALPASIM_PAI_ROOT/nre-artifacts/ego-hoods`
(section 6.2). The harmonizer checkpoint is bound read-only at NRE's default cache path
(`~/.cache/nre/harmonizer`).

---

## 14. Troubleshooting

| Symptom | Cause, and fix |
|---|---|
| `ERROR: set PROJECT in config.sh` | Set `PROJECT` in `config.sh` or export it. |
| `ImportError: Bad git executable` in `wizard.log` | GitPython can't find `git` (as on some compute nodes). The scripts set `GIT_PYTHON_REFRESH=quiet`; run through them, or export it yourself. |
| `torch.cuda.is_available()` is False | The CPU torch wheel came back, typically after a plain `uv run`/`uv sync`. Rerun `setup/create_env.sh` (step 4 reinstalls CUDA torch) and keep `UV_NO_SYNC=1`. |
| `module: command not found` in a job | Scripts re-exec themselves under `bash -l`. If your login profile doesn't define `module`, source your site's Lmod `init/bash` in `lib/common.sh`. |
| `gcc: No such file or directory` while gsplat builds | The compiler module isn't available on the compute nodes; check `ALPASIM_MODULES` there (some sites need an architecture module first). |
| Driver never becomes ready | See `runs/<RUN>/driver.log`: wrong checkpoint path, weights that would be fetched from the Hub (`HF_HUB_OFFLINE=1` on compute nodes; download them first), or out of GPU memory. |
| Renderer takes minutes to start | The MTGS renderer is indexing the trajdata cache on a cold filesystem; later starts are fast. |
| Port already in use | Another pipeline on the node uses it. Set `GPU`/`BASEPORT`/`DRIVER_PORT`. |
| `No module named 'nre.nrm'` in `renderer-0.log` | The container overlay is full (section 13). |
| `No data source found for scene_id: clipgt-...` | The scene isn't in `$ALPASIM_PAI_ROOT/nre-artifacts`: run `setup/download_pai_scenes.sh` with the same scene overrides. |
| Renderer hangs or fails loading the harmonizer | `harmonizer_nontemporal.pt` isn't in `$NRE_HARMONIZER_DIR`: `bash setup/download_checkpoints.sh harmonizer`. |
| `NRE_SIF=... not found` | Run `setup/pull_nre.sh` on a login node. |
| `apptainer pull` runs out of space | It needs about twice the image size in `$APPTAINER_TMPDIR`; point it at project storage. |
| `ERROR: the PAI ego hood is missing` | The hood images aren't in `$ALPASIM_PAI_ROOT/nre-artifacts/ego-hoods`. Run `bash setup/download_pai_scenes.sh` once (it copies them from the AlpaSim checkout), or set `EGO_HOOD=0` to run without the hood. |
| `CUDA driver version is insufficient` (PAI) | The node's NVIDIA driver is older than 570. |
| Many array tasks fail within seconds | A bad node takes task after task. Check `sacct -j <job> -X -o JobID,State,Elapsed,NodeList`, then resubmit with `--array=<indices> --exclude=<node>`. |

---

## 15. Known differences from the official evaluation

- **Renderer kernels (nuPlan).** The renderer gets the wizard's `TORCH_CUDA_ARCH_LIST`
  (`8.9;9.0+PTX`, as in the official renderer container) rather than the host's: kernels built
  natively for the GPU render slightly different images, which moves closed-loop scores. GPUs
  below compute capability 8.9 (e.g. A100) can't run those kernels, so they are built for the
  host GPU instead (automatic; `ALPASIM_KEEP_HOST_ARCH=1/0` forces either). Each arch list gets
  its own `TORCH_EXTENSIONS_DIR`, so the first run with a new list recompiles gsplat (~1 min).
- **Hardware.** Different GPUs and torch builds give small numerical differences (section 10).
  Use local scores to compare your own checkpoints, not as leaderboard numbers.
- **Scene order.** A scene's score can depend a little on which scenes the same renderer process
  ran before it, so compare full-set averages rather than single scenes.
- **Scenes.** The public `navtest` and `nurec_curated_val` sets; the leaderboard uses private sets.
- **Latency.** There is no simulated planner delay locally. The official run expects about 0.1 s
  per inference call and at most 16 GiB of VRAM per driver replica.
- **Driving command.** The adapters derive the NAVSIM-style command (left/straight/right) from
  AlpaSim's route (first waypoint ≥ 5 m ahead, ±2 m lateral threshold), which can differ from
  NAVSIM's own labels.
- **Harmonizer (PAI).** Loaded from a local copy of the official checkpoint instead of downloaded.
- **Ego hood (PAI).** Drawn on the front camera by default, as in the challenge. The PAI results
  of section 10 were scored without it (`EGO_HOOD=0`).

---

## 16. Credits and license

- [AlpaSim](https://github.com/NVlabs/alpasim) © NVIDIA, Apache-2.0. The SimScale sample drivers
  in `drivers/simscale/` are AlpaSim's (see `drivers/simscale/README.md`); the DrivoR adapter's
  gRPC, batching, trajectory and navigation modules and `tools/probe_driver.py` are adapted from
  AlpaSim's SimScale LTF sample.
- [DrivoR](https://github.com/valeoai/DrivoR) © valeo.ai; the network in
  `drivers/drivor/drivor_alpasim/original/` comes from it.
- MTGS assets and SimScale checkpoints © OpenDriveLab; NuRec scenes, the NRE image and the
  harmonizer © NVIDIA; nuPlan data © Motional. See their dataset cards and licenses for terms.
  Run the NRE image as shipped; don't try to bypass its protections.
- This repo is Apache-2.0 (see `LICENSE`).
