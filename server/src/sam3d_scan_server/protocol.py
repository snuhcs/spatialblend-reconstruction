# Copyright (c) 2026 Seoul National University
# SPDX-License-Identifier: MIT

"""Loads the generated gRPC modules for proto/distributed.proto.

The proto is in package spatialblend.scan.v1; protoc (python) still emits top-level modules named distributed_pb2 and
distributed_pb2_grpc. They are generated here, at first use, into a per-user cache keyed by the proto's
hash and by the installed protobuf and grpcio-tools versions. A generated file that the installed protobuf
runtime rejects can therefore never be checked in or go stale.
"""
from __future__ import annotations

import hashlib
import importlib
import os
import sys
from pathlib import Path

_PROTO_NAME = "distributed.proto"
_loaded = None


def proto_path() -> Path:
    """The canonical proto: packaged next to this module (wheel), else the repository's top-level proto/ directory,
    which the client library (client/, Bazel) compiles too."""
    here = Path(__file__).resolve().parent
    for candidate in (here / "proto" / _PROTO_NAME, here.parents[2] / "proto" / _PROTO_NAME):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{_PROTO_NAME} not found beside the package or in the repository's proto/")


def _cache_root() -> Path:
    base = os.environ.get("SAM3D_SCAN_CACHE") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "sam3d-scan")
    return Path(base) / "pb"


def _versions() -> str:
    import google.protobuf
    from grpc_tools import protoc  # noqa: F401  (import proves it is installed)
    try:
        import importlib.metadata as md
        tools = md.version("grpcio-tools")
    except Exception:
        tools = "unknown"
    return f"{google.protobuf.__version__}-{tools}"


def load():
    """Returns (distributed_pb2, distributed_pb2_grpc)."""
    global _loaded
    if _loaded is not None:
        return _loaded

    proto = proto_path()
    key = hashlib.sha256(proto.read_bytes() + _versions().encode()).hexdigest()[:16]
    out = _cache_root() / key
    if not (out / "distributed_pb2_grpc.py").is_file():
        from grpc_tools import protoc
        out.mkdir(parents=True, exist_ok=True)
        rc = protoc.main(["protoc", f"-I{proto.parent}", f"--python_out={out}", f"--grpc_python_out={out}",
                          str(proto)])
        if rc != 0:
            raise RuntimeError(f"protoc failed for {proto} (exit {rc})")
    if str(out) not in sys.path:
        sys.path.insert(0, str(out))
    pb2 = importlib.import_module("distributed_pb2")
    pb2_grpc = importlib.import_module("distributed_pb2_grpc")
    _loaded = (pb2, pb2_grpc)
    return _loaded
