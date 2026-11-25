# Copyright (c) 2026 Seoul National University
# SPDX-License-Identifier: MIT

"""Server configuration: where the SAM 3D Objects checkout and checkpoints are, which GPUs, which CUDA arch.

Every setting has a command-line flag and an environment variable (the flag wins). Nothing here is specific to one
machine: the defaults are a loopback bind on port 9800, GPU 0, and the checkpoints inside the SAM 3D Objects checkout.

    SAM3D_SCAN_HOST              bind address (default 127.0.0.1; use 0.0.0.0 or [::] to listen on every interface)
    SAM3D_SCAN_PORT              listen port (default 9800)
    SAM3D_SCAN_GPUS              physical GPU indices, comma separated (default 0); sets CUDA_VISIBLE_DEVICES
    SAM3D_OBJECTS_ROOT           checkout of facebookresearch/sam-3d-objects (holds notebook/inference.py)
    SAM3D_OBJECTS_CHECKPOINTS    directory holding pipeline.yaml and the checkpoints (default <root>/checkpoints/hf)
    SAM3D_SCAN_DEBUG_DIR         when set, a mask overlay PNG per Segment request is written here
    SAM3D_SCAN_MAX_MESSAGE_MB    gRPC message limit each way (default 100)
    SAM3D_SCAN_REQUEST_TIMEOUT_S how long one request may wait for its worker (default 600)
    SAM3D_SCAN_LOG_LEVEL         Python logging level (default INFO)
    TORCH_CUDA_ARCH_LIST         CUDA arch for JIT-built extensions (default: detected with nvidia-smi)
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Mapping, Optional

DEFAULT_PORT = 9800
DEFAULT_HOST = "127.0.0.1"
SAM3_REPO_ID = "facebook/sam3"
SAM3D_OBJECTS_REPO_ID = "facebook/sam-3d-objects"


@dataclass
class ServerConfig:
    port: int = DEFAULT_PORT
    host: str = DEFAULT_HOST
    # Physical GPU indices. Workers address them as cuda:0.. after CUDA_VISIBLE_DEVICES is set from this list.
    gpus: List[int] = field(default_factory=lambda: [0])
    # Checkout of facebookresearch/sam-3d-objects: holds notebook/inference.py (and, by default, checkpoints/hf).
    sam3d_root: Optional[Path] = None
    # Directory with pipeline.yaml and the SAM 3D Objects checkpoints. None: <sam3d_root>/checkpoints/hf.
    checkpoints: Optional[Path] = None
    cuda_arch: Optional[str] = None
    # When set, SAM3 writes a mask overlay PNG per request here. Off by default.
    debug_dir: Optional[Path] = None
    stub: bool = False
    max_message_mb: int = 100
    max_workers: int = 20
    request_timeout_s: float = 600.0

    @property
    def sam3d_pipeline_yaml(self) -> Path:
        return self.checkpoints_dir / "pipeline.yaml"

    @property
    def checkpoints_dir(self) -> Path:
        if self.checkpoints is not None:
            return Path(self.checkpoints)
        return (self.sam3d_root or Path(".")) / "checkpoints" / "hf"


def parse_gpus(text: str) -> List[int]:
    """'0', '0,1' or '0 1' -> [0, 1]. Raises ValueError on anything else."""
    items = [t for t in text.replace(",", " ").split() if t]
    if not items:
        raise ValueError("no GPU index given")
    return [int(t) for t in items]


def env_defaults(env: Optional[Mapping[str, str]] = None, skip=()) -> dict:
    """The ServerConfig fields the environment sets (only the ones present and not in skip, the fields a flag
    already gives). Raises ValueError on a malformed value."""
    env = os.environ if env is None else env
    env = {k: v for k, v in env.items() if _FIELD_OF.get(k) not in skip}
    out: dict = {}
    if env.get("SAM3D_SCAN_HOST"):
        out["host"] = env["SAM3D_SCAN_HOST"]
    if env.get("SAM3D_SCAN_PORT"):
        out["port"] = int(env["SAM3D_SCAN_PORT"])
    if env.get("SAM3D_SCAN_GPUS"):
        out["gpus"] = parse_gpus(env["SAM3D_SCAN_GPUS"])
    if env.get("SAM3D_OBJECTS_ROOT"):
        out["sam3d_root"] = Path(env["SAM3D_OBJECTS_ROOT"]).expanduser()
    if env.get("SAM3D_OBJECTS_CHECKPOINTS"):
        out["checkpoints"] = Path(env["SAM3D_OBJECTS_CHECKPOINTS"]).expanduser()
    if env.get("SAM3D_SCAN_DEBUG_DIR"):
        out["debug_dir"] = Path(env["SAM3D_SCAN_DEBUG_DIR"]).expanduser()
    if env.get("SAM3D_SCAN_MAX_MESSAGE_MB"):
        out["max_message_mb"] = int(env["SAM3D_SCAN_MAX_MESSAGE_MB"])
    if env.get("SAM3D_SCAN_REQUEST_TIMEOUT_S"):
        out["request_timeout_s"] = float(env["SAM3D_SCAN_REQUEST_TIMEOUT_S"])
    return out


_FIELD_OF = {"SAM3D_SCAN_HOST": "host", "SAM3D_SCAN_PORT": "port", "SAM3D_SCAN_GPUS": "gpus",
             "SAM3D_OBJECTS_ROOT": "sam3d_root", "SAM3D_OBJECTS_CHECKPOINTS": "checkpoints",
             "SAM3D_SCAN_DEBUG_DIR": "debug_dir", "SAM3D_SCAN_MAX_MESSAGE_MB": "max_message_mb",
             "SAM3D_SCAN_REQUEST_TIMEOUT_S": "request_timeout_s"}


def detect_cuda_arch(gpu_index: int) -> Optional[str]:
    """Compute capability of one GPU as TORCH_CUDA_ARCH_LIST text ("8.6"), from nvidia-smi, no torch import."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None
    try:
        out = subprocess.run([smi, "-i", str(gpu_index), "--query-gpu=compute_cap", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20, check=True).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None
    first = out.splitlines()[0].strip() if out else ""
    return first if first and first[0].isdigit() else None


def resolve_cuda_arch(cfg: ServerConfig) -> Optional[str]:
    """--cuda-arch, else TORCH_CUDA_ARCH_LIST, else the first selected GPU's capability. None leaves torch to decide."""
    if cfg.cuda_arch:
        return cfg.cuda_arch
    env = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if env:
        return env
    return detect_cuda_arch(cfg.gpus[0]) if cfg.gpus else None


def apply_environment(cfg: ServerConfig) -> None:
    """Sets the variables sam3d_objects and torch read at import time. Call before importing either."""
    os.environ["LIDRA_SKIP_INIT"] = "1"
    # --gpu N is a physical index as nvidia-smi numbers it; CUDA's default order may differ.
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if cfg.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in cfg.gpus)
    arch = resolve_cuda_arch(cfg)
    if arch:
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
