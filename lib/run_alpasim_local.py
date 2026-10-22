#!/usr/bin/env python
"""Run an AlpaSim wizard output directory without containers.

The wizard (with wizard.run_method=NONE) writes <log_dir>/docker-compose.yaml.
This script replays it on the host: container mount points (/mnt/..., /repo/...)
in each service's command are rewritten to the host paths from its volumes, and
every service runs as a plain process in the current (alpasim) env. The one
exception is a service from a prebuilt image (the closed-source NRE renderer of
the PAI track), which runs under Apptainer (or SingularityCE) from $NRE_SIF. Servers are
started first; the runtime runs in the foreground; everything is stopped when
the runtime exits. Your driver must already be listening (see network config).

    python run_alpasim_local.py <log_dir>
"""
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import yaml

log_dir = Path(sys.argv[1]).resolve()
compose = yaml.safe_load((log_dir / "docker-compose.yaml").read_text())
txt_logs = log_dir / "txt-logs"
txt_logs.mkdir(exist_ok=True)
repo = os.environ.get("ALPASIM_REPO")


def host_command(svc: dict) -> str:
    cmd = svc["command"][-1]
    # longest container path first so /mnt/log_dir wins over /mnt
    mounts = sorted(
        (v.split(":", 1)[::-1] for v in svc.get("volumes", [])),
        key=lambda m: -len(m[0]),
    )
    for container, host in mounts:
        cmd = cmd.replace(container, host)
    # the generated configs the command points to hold container paths too
    # (e.g. the runtime's usdz data_dir=/mnt/nre-data/... on the PAI track)
    for tok in cmd.replace("=", " ").split():
        cfg = Path(tok)
        if tok.endswith((".yaml", ".yml")) and cfg.is_file() and log_dir in cfg.resolve().parents:
            text = original = cfg.read_text()
            for container, host in mounts:
                text = text.replace(container, host)
            if text != original:
                cfg.write_text(text)
    # the env already holds everything; `uv run` would try to re-sync it
    return cmd.replace("uv run ", "")


def wait_port(port: int, proc: subprocess.Popen, name: str, timeout: float = 900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            sys.exit(f"{name} exited early (code {proc.returncode}); see {txt_logs}/{name}.log")
        with socket.socket() as s:
            if s.connect_ex(("localhost", port)) == 0:
                print(f"   {name} up on :{port} ({time.time() - t0:.0f}s)", flush=True)
                return
        time.sleep(2)
    sys.exit(f"{name} did not open :{port} within {timeout:.0f}s")


def container_runtime() -> str:
    """Apptainer, or SingularityCE where Apptainer isn't installed (some clusters have
    only `singularity`). Both take the same exec flags. CONTAINER_RUNTIME overrides the
    choice."""
    runtime = (os.environ.get("CONTAINER_RUNTIME")
               or shutil.which("apptainer") or shutil.which("singularity"))
    if not runtime:
        sys.exit("neither apptainer nor singularity is on PATH; set CONTAINER_RUNTIME")
    return runtime


def container_command(svc: dict, gpus: list | None) -> list:
    """Apptainer command for a service that runs from a prebuilt image (no `build`
    key: the NRE renderer of the PAI track). It runs as in Docker: the image's own
    env and working dir, the compose volumes bound at their container paths, the
    command unchanged, host network."""
    image = svc["image"]
    if not image.startswith("nvcr.io/nvidia/nre/"):
        sys.exit(f"no container image for {image}; only the NRE renderer runs under Apptainer")
    sif = os.environ.get("NRE_SIF", "")
    if not os.path.isfile(sif):
        sys.exit(f"NRE_SIF={sif!r} not found; build it with setup/pull_nre.sh")
    if os.environ.get("NRE_IMAGE") != image:
        print(f"WARNING: compose wants {image}, running {sif} (NRE_IMAGE={os.environ.get('NRE_IMAGE')})",
              flush=True)
    # --cleanenv: the host's conda/module variables must not reach the image's Python
    # and CUDA; --no-home: NRE writes caches to $HOME, kept inside the container.
    args = [container_runtime(), "exec", "--nv", "--cleanenv", "--no-home", "--writable-tmpfs",
            "--pwd", svc.get("working_dir", os.environ.get("NRE_WORKDIR", "/app/run.runfiles/_main"))]
    for v in svc.get("volumes", []):
        os.makedirs(v.split(":")[0], exist_ok=True)  # Docker creates missing bind sources too
        args += ["--bind", v]  # host:container[:ro], same syntax as compose
    # The harmonizer checkpoint goes where NRE looks for it by default
    # (~/.cache/nre/harmonizer), so the renderer command stays the official one.
    harmonizer = os.environ.get("NRE_HARMONIZER_DIR", "")
    if os.path.isfile(os.path.join(harmonizer, "harmonizer_nontemporal.pt")):
        args += ["--bind", f"{harmonizer}:{os.path.expanduser('~')}/.cache/nre/harmonizer:ro"]
    elif any("--enable-harmonizer" in c for c in svc["command"]):
        print(f"WARNING: no harmonizer_nontemporal.pt in NRE_HARMONIZER_DIR={harmonizer!r}; "
              "NRE will try to download it", flush=True)
    # Python writes .pyc files into the image tree on import; Docker's writable layer
    # holds them, but --writable-tmpfs is capped (often 64 MB) and fills up, which
    # NRE's obfuscated loader reports as "No module named 'nre.nrm'".
    args += ["--env", "PYTHONDONTWRITEBYTECODE=1"]
    for e in svc.get("environment", []):
        args += ["--env", e]
    if gpus:
        args += ["--env", f"CUDA_VISIBLE_DEVICES={','.join(gpus)}"]
    return args + [sif, svc.get("entrypoint", "bash"), *svc["command"]]


def keep_host_arch() -> bool:
    """Build gsplat for this GPU instead of the wizard's arch list ("8.9;9.0+PTX")?
    That list has no kernel a GPU below compute capability 8.9 (e.g. an A100, sm_80) can
    run, so those GPUs always use the host's list. ALPASIM_KEEP_HOST_ARCH=1/0 forces it."""
    flag = os.environ.get("ALPASIM_KEEP_HOST_ARCH")
    if flag is not None:
        return flag == "1"
    try:
        cap = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout.split()[0]
        return float(cap) < 8.9
    except (OSError, subprocess.CalledProcessError, IndexError, ValueError):
        return False


def start(name: str, svc: dict) -> subprocess.Popen:
    gpus = (svc.get("deploy", {}).get("resources", {}).get("reservations", {})
            .get("devices", [{}])[0].get("device_ids"))
    if "build" not in svc:
        argv = container_command(svc, gpus)
        print(f">> {name} ({os.path.basename(argv[0])}): {svc['command'][-1].splitlines()[-1]}", flush=True)
        log = open(txt_logs / f"{name}.log", "w")
        # Keep host-injected preloads and binds (e.g. an XALT job tracker's
        # SINGULARITYENV_LD_PRELOAD) out of the image: it runs as in Docker, without them.
        env = {k: v for k, v in os.environ.items()
               if k not in ("SINGULARITYENV_LD_PRELOAD", "APPTAINERENV_LD_PRELOAD",
                            "SINGULARITY_BINDPATH", "APPTAINER_BINDPATH")}
        return subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)

    cmd = host_command(svc)
    env = dict(os.environ)
    for e in svc.get("environment", []):
        k, v = e.split("=", 1)
        env[k] = v
    # Use the wizard's TORCH_CUDA_ARCH_LIST (the official renderer builds gsplat for
    # "8.9;9.0+PTX"): on GPUs outside that list the kernels are JIT-compiled from PTX,
    # and building natively for the GPU instead changes the renderings enough to move
    # closed-loop scores. torch's JIT cache is not keyed on the arch list, so give each
    # arch list its own extensions dir. The host's list is used instead on GPUs that list
    # can't run (keep_host_arch).
    if keep_host_arch() and "TORCH_CUDA_ARCH_LIST" in os.environ:
        env["TORCH_CUDA_ARCH_LIST"] = os.environ["TORCH_CUDA_ARCH_LIST"]
    if "TORCH_CUDA_ARCH_LIST" in env:
        base = env.get("TORCH_EXTENSIONS_DIR", os.path.expanduser("~/.cache/torch_extensions"))
        slug = "".join(c if c.isalnum() else "_" for c in env["TORCH_CUDA_ARCH_LIST"])
        env["TORCH_EXTENSIONS_DIR"] = os.path.join(base, f"arch_{slug}")
    if gpus:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    print(f">> {name}: {cmd.splitlines()[-1]}", flush=True)
    log = open(txt_logs / f"{name}.log", "w")
    return subprocess.Popen(["bash", "-c", cmd], env=env, cwd=repo, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)


procs = {}
try:
    servers = {n: s for n, s in compose["services"].items() if not n.startswith("runtime")}
    for name, svc in servers.items():
        procs[name] = start(name, svc)
    for name, svc in servers.items():
        port = int(str(svc["ports"][0]).split(":")[0])
        wait_port(port, procs[name], name)

    rc = 0
    for name, svc in compose["services"].items():
        if name.startswith("runtime"):
            p = start(name, svc)
            procs[name] = p
            rc = rc or p.wait()
            print(f">> {name} exited with {p.returncode}", flush=True)
finally:
    for name, p in procs.items():
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGTERM)
    time.sleep(3)
    for p in procs.values():
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)

summary = log_dir / "aggregate" / "results-summary.json"
print(f">> results: {summary if summary.exists() else 'none (see txt-logs/)'}")
sys.exit(rc)
