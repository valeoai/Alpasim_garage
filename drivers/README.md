# Drivers

A **driver** is the policy under test. AlpaSim talks to it over gRPC
(`egodriver.EgodriverService`, defined in
`$ALPASIM_REPO/src/grpc/alpasim_grpc/v0/egodriver.proto`). Every `*.sh` file
here starts one driver in the foreground.

## Script contract

`run_eval.sh` (and each SLURM shard) runs `bash drivers/<name>.sh`, or a path
you pass instead, with the modules loaded, the `alpasim` env active, and:

| Variable | Meaning |
|---|---|
| `ALPASIM_DRIVER_PORT` | port to listen on (bind `0.0.0.0` or `localhost`) |
| `CUDA_VISIBLE_DEVICES` | the one GPU to use |
| `ALPASIM_DRIVER_LOG_DIR` | a writable folder for the driver's own logs |
| `DRIVER_MAX_BATCH_SIZE` | suggested max batch size (default 4; up to 4 rollouts run concurrently per pipeline) |
| `HF_HUB_OFFLINE=1` | compute nodes have no internet, so all weights must be local |

The script must:

- keep running until it is killed (SIGTERM);
- answer `get_version` **only once the model is loaded**, and return
  `UNAVAILABLE` before that. `tools/wait_for_driver.py` polls `get_version` and
  starts the simulation on the first success;
- start with `driver_setup` from `lib/common.sh` (see the existing scripts), so
  it also works when you start it by hand.

## RPCs a driver must handle

| RPC | Input | What to do |
|---|---|---|
| `start_session` / `close_session` | one session per rollout; lists the available cameras and intrinsics | keep per-session state |
| `submit_image_observation` | JPEG of one camera (`CAM_F0, L0, L1, L2, R0, R1, R2, B0`, 1920×1080) with frame timestamps | decode and store the latest per camera |
| `submit_egomotion_observation` | ego pose history in the scene's `local` frame, plus velocity and acceleration | store the latest |
| `submit_route` | route waypoints in the ego (rig) frame, starting ~40 m ahead | e.g. derive a left/straight/right command |
| `submit_recording_ground_truth` | not sent on the nuPlan track | ignore |
| `drive` | `time_now_us`, `time_query_us` | return a `Trajectory` |
| `get_version` | – | version ID, once ready |

### What `drive` returns

A `DriveResponse` with `trajectory.poses`: a list of `PoseAtTime`.

- `timestamp_us`: absolute sim time. The first pose is at `time_now_us`, and
  the last must be at or after `time_query_us`.
- `pose.vec` (x, y, z) and `pose.quat` (w, x, y, z): in the scene's **`local`**
  world frame, not relative to the ego.

A NAVSIM-style model predicts a rig-frame `(N, 3)` array of (x, y, yaw), e.g. 8
points at 0.5 s. To convert it: take the latest ego pose `(X, Y, Z, ψ)` in
`local`, then for each point

```
X' = X + x·cos ψ − y·sin ψ,   Y' = Y + x·sin ψ + y·cos ψ,   yaw' = ψ + θ,   z = Z
```

Use timestamps `time_now_us + k·500_000`, and put the current pose first.
`drivor/drivor_alpasim/trajectory.py` does exactly this, including resampling
to 10 Hz.

## Adding a NAVSIM model (recommended route)

Copy the DrivoR adapter. Only two files are model-specific:

```
cp -r drivers/drivor drivers/mymodel
mv drivers/mymodel/drivor_alpasim drivers/mymodel/mymodel_alpasim
cp drivers/drivor.sh drivers/mymodel.sh      # edit: package name, env vars
```

| File | Model-specific? | Role |
|---|---|---|
| `policy.py` | **yes** | build the model, load the checkpoint, `predict_batch(list[InferenceInput]) -> list[Prediction]` with rig-frame `(8, 3)` trajectories |
| `preprocessing.py` | **yes** | camera selection and order, resize, ego-status vector |
| `driver.py` | rename only | gRPC service, sessions, camera/pose sync, counters |
| `batch_worker.py` | no | batches concurrent `drive` calls onto the GPU |
| `trajectory.py` | no | rig → local conversion, 10 Hz output, straight-line fallback |
| `navigation.py` | no | route → one-hot command (first waypoint ≥ 5 m ahead, ±2 m lateral threshold) |

Before running the simulator, test the adapter against a live driver:

```bash
bash drivers/mymodel.sh &                       # or ALPASIM_DRIVER_PORT=7100 ...
python tools/wait_for_driver.py localhost:6789
python tools/probe_driver.py --address localhost:6789
kill %1
bash run_eval.sh mymodel dev
```

Check that the driver log's counters show `*_inference` equal to the number of
decisions, and `straight_fallback=0, inference_error=0`.

## DrivoR specifics

- **Model:** the upstream DrivoR network ([valeoai/DrivoR](https://github.com/valeoai/DrivoR)),
  copied into `drivor/drivor_alpasim/original/` and built from its agent config with the
  `agent.config.*` overrides of its NAVSIM navtest evaluation (`CONFIG_OVERRIDES` in `policy.py`).
- **Checkpoint:** the released `drivor_Nav1_25epochs.pth` (`setup/download_checkpoints.sh drivor`),
  loaded with `strict=True`. Only the model is imported, not the NAVSIM agent, so neither
  NAVSIM nor the nuPlan devkit is needed.
- **Inputs:** `CAM_F0, CAM_B0, CAM_L0, CAM_R0`, resized to 1148×672 with PIL and
  ImageNet-normalized, plus the ego status `[0,0,0, vx,vy, ax,ay, command(4)]`, as NAVSIM's
  feature builder makes them.
- **PAI track:** the four nuPlan views are synthesized from the PAI f-theta cameras
  (`drivor/drivor_alpasim/pai_views.py`; the SimScale drivers use the same module).
- **Cost:** about 120 ms per step (60 ms preprocessing + 58 ms model) and 0.8 GiB
  of GPU memory.
