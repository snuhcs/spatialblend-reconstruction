# Copyright (c) 2026 Seoul National University
# SPDX-License-Identifier: MIT

"""sam3d-scan-server: run, health-check, or stub.

    sam3d-scan-server --gpu 0 --sam3d-root ~/sam-3d-objects        # listens on 127.0.0.1:9800
    sam3d-scan-server --host 0.0.0.0 --port 9800 ...                # every interface (put TLS in front, see README)
    sam3d-scan-server health --server localhost:9800 --wait 600     # exit 0 once every model is loaded
    sam3d-scan-server --stub                                        # CPU-only deterministic stand-in, no models

Every flag has an environment variable (see sam3d_scan_server.config); the flag wins.
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
from pathlib import Path
from typing import List, Optional

from .config import ServerConfig, apply_environment, env_defaults

EXIT_OK, EXIT_UNHEALTHY, EXIT_USAGE = 0, 1, 2


def _serve_parser(p: argparse.ArgumentParser) -> None:
    p.add_argument("--gpu", type=int, nargs="+", default=None, metavar="N",
                   help="physical GPU index (several allowed). Sets CUDA_VISIBLE_DEVICES. Env SAM3D_SCAN_GPUS. "
                        "Default 0.")
    p.add_argument("--port", type=int, default=None, help="listen port. Env SAM3D_SCAN_PORT. Default 9800.")
    p.add_argument("--host", default=None,
                   help="bind address. Env SAM3D_SCAN_HOST. Default 127.0.0.1 (loopback only).")
    p.add_argument("--sam3d-root", type=Path, default=None,
                   help="checkout of facebookresearch/sam-3d-objects. Env SAM3D_OBJECTS_ROOT.")
    p.add_argument("--checkpoints", type=Path, default=None,
                   help="directory with pipeline.yaml and the SAM 3D Objects checkpoints. "
                        "Env SAM3D_OBJECTS_CHECKPOINTS. Default <sam3d-root>/checkpoints/hf.")
    p.add_argument("--cuda-arch", default=None,
                   help="TORCH_CUDA_ARCH_LIST, for example 8.6 or 8.9. Default: env, else nvidia-smi.")
    p.add_argument("--debug-dir", type=Path, default=None,
                   help="write a mask overlay PNG per segment request here. Env SAM3D_SCAN_DEBUG_DIR.")
    p.add_argument("--stub", action="store_true", help="serve the deterministic CPU stub, load no models")
    p.add_argument("--log-level", default=None, help="Env SAM3D_SCAN_LOG_LEVEL. Default INFO.")


def _health_parser(p: argparse.ArgumentParser) -> None:
    p.add_argument("--server", required=True, metavar="HOST:PORT")
    p.add_argument("--timeout", type=float, default=5.0, help="seconds (default 5)")
    p.add_argument("--wait", type=float, default=0.0,
                   help="keep retrying for up to this many seconds until the server is SERVING (one bounded loop)")


def check_health(address: str, timeout: float, wait: float = 0.0) -> int:
    """0 SERVING, 1 not serving or unreachable. Prints one line."""
    import time

    import grpc
    from grpc_health.v1 import health_pb2, health_pb2_grpc

    deadline = time.monotonic() + max(wait, 0.0)
    detail = "unreachable"
    while True:
        try:
            with grpc.insecure_channel(address) as channel:
                reply = health_pb2_grpc.HealthStub(channel).Check(
                    health_pb2.HealthCheckRequest(service=""), timeout=timeout)
            detail = health_pb2.HealthCheckResponse.ServingStatus.Name(reply.status)
            if reply.status == health_pb2.HealthCheckResponse.SERVING:
                print(f"{address}: SERVING")
                return EXIT_OK
        except grpc.RpcError as e:
            detail = f"unreachable ({e.code().name})"
        if time.monotonic() >= deadline:
            print(f"{address}: {detail}")
            return EXIT_UNHEALTHY
        time.sleep(1.0)


def build_config(args: argparse.Namespace, env=None) -> ServerConfig:
    """Environment first, then the flags given on the command line."""
    flags = {"port": args.port, "host": args.host, "gpus": args.gpu, "sam3d_root": args.sam3d_root,
             "checkpoints": args.checkpoints, "cuda_arch": args.cuda_arch, "debug_dir": args.debug_dir}
    flags = {k: (v.expanduser() if isinstance(v, Path) else v) for k, v in flags.items() if v is not None}
    fields = env_defaults(env, skip=set(flags))
    fields.update(flags)
    return ServerConfig(stub=args.stub, **fields)


def run_server(args: argparse.Namespace) -> int:
    level = args.log_level or os.environ.get("SAM3D_SCAN_LOG_LEVEL") or "INFO"
    logging.basicConfig(level=getattr(logging, str(level).upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(message)s")
    try:
        cfg = build_config(args)
    except ValueError as e:
        print(f"error: bad configuration in the environment: {e}", file=sys.stderr)
        return EXIT_USAGE
    if not cfg.stub:
        if cfg.sam3d_root is None:
            print("error: --sam3d-root (or SAM3D_OBJECTS_ROOT) is required unless --stub", file=sys.stderr)
            return EXIT_USAGE
        if not cfg.sam3d_pipeline_yaml.is_file():
            print(f"error: {cfg.sam3d_pipeline_yaml} not found. Download the checkpoints with sam3d-scan-models "
                  "(see README.md).", file=sys.stderr)
            return EXIT_USAGE
        apply_environment(cfg)  # before torch or sam3d_objects is imported

    from .server import build_server

    server = build_server(cfg)
    stop_requested = threading.Event()

    def _on_signal(signum, _frame):
        logging.getLogger("sam3d_scan_server").info("signal %s: shutting down", signum)
        stop_requested.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    server.start()
    try:
        while not stop_requested.wait(0.5):
            pass
    finally:
        server.stop(grace=5.0)
    return EXIT_OK


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "health":
        p = argparse.ArgumentParser(prog="sam3d-scan-server health")
        _health_parser(p)
        a = p.parse_args(argv[1:])
        return check_health(a.server, a.timeout, a.wait)
    p = argparse.ArgumentParser(prog="sam3d-scan-server", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    _serve_parser(p)
    return run_server(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
