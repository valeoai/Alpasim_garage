#!/usr/bin/env python
"""Wait until an AlpaSim driver answers get_version, i.e. is ready to drive.

    python tools/wait_for_driver.py localhost:6789 [--timeout 900] [--pid PID]

Model drivers answer UNAVAILABLE while the checkpoint is still loading, so this
keeps polling until the call succeeds. Exits 1 on timeout, or as soon as the
driver process (--pid) has died.
"""
import argparse
import os
import sys
import time

import grpc
from alpasim_grpc.v0 import common_pb2, egodriver_pb2_grpc

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("address")
parser.add_argument("--timeout", type=float, default=900.0)
parser.add_argument("--pid", type=int, help="driver process; stop waiting if it dies")
args = parser.parse_args()


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


t0 = time.time()
last_error = "no attempt yet"
with grpc.insecure_channel(args.address) as channel:
    stub = egodriver_pb2_grpc.EgodriverServiceStub(channel)
    while time.time() - t0 < args.timeout:
        if args.pid and not alive(args.pid):
            sys.exit(f"driver process {args.pid} exited before becoming ready")
        try:
            version = stub.get_version(common_pb2.Empty(), timeout=10)
            print(f"driver ready after {time.time() - t0:.0f}s: "
                  f"{version.version_id} (git {version.git_hash})", flush=True)
            sys.exit(0)
        except grpc.RpcError as error:
            last_error = f"{error.code().name}: {error.details()}"
        time.sleep(5)
sys.exit(f"driver at {args.address} not ready after {args.timeout:.0f}s ({last_error})")
