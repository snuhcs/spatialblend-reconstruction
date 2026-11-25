# Copyright (c) 2026 Seoul National University
# SPDX-License-Identifier: MIT

"""CPU-only smoke test. No GPU, no torch, no checkpoints.

It proves the parts that do not need a model: every module compiles and imports, the proto loads, the servicer
queues work to fake workers and answers, a pool whose workers all failed to load fails requests at once instead
of hanging them, the gRPC health service flips with readiness, the CLI health check agrees, and the server
shuts down cleanly.
"""
import io
import py_compile
import queue
import socket
import sys
import threading
from pathlib import Path

import grpc
import numpy as np
import pytest
from grpc_health.v1 import health_pb2, health_pb2_grpc
from PIL import Image

SRC = Path(__file__).resolve().parents[1] / "src" / "sam3d_scan_server"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def png(w=8, h=6, color=(200, 30, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


def test_every_module_compiles():
    for path in sorted(SRC.glob("*.py")):
        py_compile.compile(str(path), doraise=True)


def test_heavy_modules_are_not_imported_at_import_time():
    import sam3d_scan_server.server  # noqa: F401
    import sam3d_scan_server.cli  # noqa: F401
    import sam3d_scan_server.models  # noqa: F401
    for name in ("torch", "pytorch3d", "transformers", "sam3d_objects"):
        assert name not in sys.modules, f"{name} was imported at module import time"


def test_proto_loads_and_roundtrips():
    from sam3d_scan_server import protocol
    pb, pb_grpc = protocol.load()
    req = pb.SegmentRequest(client_id="c", image_data=b"x", points=[pb.SegPoint(x=1, y=2, z=0)])
    assert pb.SegmentRequest.FromString(req.SerializeToString()).points[0].y == 2
    assert hasattr(pb_grpc, "ReconstructionServiceStub")


def test_cuda_arch_resolution(monkeypatch):
    from sam3d_scan_server.config import ServerConfig, resolve_cuda_arch
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
    assert resolve_cuda_arch(ServerConfig(cuda_arch="8.9")) == "8.9"
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "9.0")
    assert resolve_cuda_arch(ServerConfig()) == "9.0"
    assert resolve_cuda_arch(ServerConfig(cuda_arch="8.6")) == "8.6"  # the flag wins over the environment


def test_valid_points_in_bounds():
    from sam3d_scan_server.server import pb, valid_points_in_bounds
    pts = [pb.SegPoint(x=1, y=1), pb.SegPoint(x=50, y=1), pb.SegPoint(x=-1, y=2)]
    assert valid_points_in_bounds(pts, 10, 10) == ([[1.0, 1.0]], [1])


class FakeWorker(threading.Thread):
    """Stands in for a model worker: answers every job with a canned result."""

    def __init__(self, kind, gpu_id, request_queue, response_queues, cfg, load_ok=True):
        super().__init__(daemon=True)
        self.kind, self.q, self.responses = kind, request_queue, response_queues
        self.model_ready = load_ok
        self.ready = threading.Event()
        if load_ok:
            self.ready.set()

    def run(self):
        while self.model_ready:
            job = self.q.get()
            req_id = job[0]
            if self.kind == "seg":
                mask = np.ones((4, 5), dtype=np.uint8)
                result = {"success": True, "message": "ok", "mask_data": mask.tobytes(),
                          "mask_width": 5, "mask_height": 4}
            else:
                result = {"success": True, "message": "ok", "glb_data": b"glTF", "json_data": b"{}"}
            self.responses[req_id].put(result)
            self.q.task_done()


@pytest.fixture
def served():
    from sam3d_scan_server.config import ServerConfig
    from sam3d_scan_server.server import ReconstructionServicer, ScanServer

    cfg = ServerConfig(port=free_port(), host="127.0.0.1", gpus=[0], request_timeout_s=5.0)
    servicer = ReconstructionServicer(
        cfg,
        sam3_factory=lambda g, q, r, c: FakeWorker("seg", g, q, r, c),
        sam3d_object_factory=lambda g, q, r, c: FakeWorker("obj", g, q, r, c))
    server = ScanServer(cfg, servicer)
    server.start()
    yield cfg, server
    server.stop(grace=1.0)


def test_servicer_roundtrip_through_grpc(served):
    from sam3d_scan_server import protocol
    pb, pb_grpc = protocol.load()
    cfg, _ = served
    with grpc.insecure_channel(f"127.0.0.1:{cfg.port}") as ch:
        stub = pb_grpc.ReconstructionServiceStub(ch)
        seg = stub.Segment(pb.SegmentRequest(client_id="t", image_data=png(),
                                             points=[pb.SegPoint(x=1, y=1)]), timeout=10)
        assert seg.success and (seg.mask_width, seg.mask_height) == (5, 4) and len(seg.mask_data) == 20
        rec = stub.Reconstruct(pb.ReconstructRequest(client_id="t", image_data=png(), mask_data=seg.mask_data,
                                                     mask_width=5, mask_height=4), timeout=10)
        assert rec.success and rec.glb_file == b"glTF" and rec.json_file == b"{}"


def test_health_serving_and_cli_check(served, capsys):
    from sam3d_scan_server.cli import check_health
    cfg, _ = served
    assert check_health(f"127.0.0.1:{cfg.port}", timeout=5.0, wait=10.0) == 0
    assert "SERVING" in capsys.readouterr().out


def test_unreachable_server_is_unhealthy(capsys):
    from sam3d_scan_server.cli import check_health
    assert check_health(f"127.0.0.1:{free_port()}", timeout=1.0) == 1
    assert "unreachable" in capsys.readouterr().out


def test_pool_with_no_loadable_worker_fails_at_once_not_hangs():
    from sam3d_scan_server import protocol
    from sam3d_scan_server.config import ServerConfig
    from sam3d_scan_server.server import ReconstructionServicer, ScanServer
    pb, pb_grpc = protocol.load()

    cfg = ServerConfig(port=free_port(), host="127.0.0.1", gpus=[0], request_timeout_s=5.0)
    servicer = ReconstructionServicer(
        cfg,
        sam3_factory=lambda g, q, r, c: FakeWorker("seg", g, q, r, c, load_ok=False),
        sam3d_object_factory=lambda g, q, r, c: FakeWorker("obj", g, q, r, c))
    server = ScanServer(cfg, servicer)
    server.start()
    try:
        assert servicer.readiness()["serving"] is False
        with grpc.insecure_channel(f"127.0.0.1:{cfg.port}") as ch:
            stub = pb_grpc.ReconstructionServiceStub(ch)
            r = stub.Segment(pb.SegmentRequest(client_id="t", image_data=png(), points=[pb.SegPoint(x=1, y=1)]),
                             timeout=10)
            assert not r.success and "no worker" in r.message
            hs = health_pb2_grpc.HealthStub(ch)
            import time
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                st = hs.Check(health_pb2.HealthCheckRequest(service="")).status
                if st == health_pb2.HealthCheckResponse.NOT_SERVING:
                    break
                time.sleep(0.1)
            assert st == health_pb2.HealthCheckResponse.NOT_SERVING
    finally:
        server.stop(grace=1.0)


def test_shutdown_is_clean_and_idempotent(served):
    cfg, server = served
    server.stop(grace=1.0)
    server.stop(grace=1.0)  # a second call is a no-op
    with pytest.raises(Exception):
        with grpc.insecure_channel(f"127.0.0.1:{cfg.port}") as ch:
            grpc.channel_ready_future(ch).result(timeout=1.0)


def test_stub_server_speaks_the_protocol():
    """The stub is the CPU stand-in the client selftest uses: segment then reconstruct."""
    import subprocess
    from sam3d_scan_server import protocol
    from sam3d_scan_server.cli import check_health
    pb, pb_grpc = protocol.load()
    port = free_port()
    proc = subprocess.Popen([sys.executable, "-m", "sam3d_scan_server", "--stub", "--host", "127.0.0.1",
                             "--port", str(port)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        assert check_health(f"127.0.0.1:{port}", timeout=2.0, wait=30.0) == 0
        img = Image.new("RGB", (40, 30), (10, 10, 10))
        for x in range(10, 30):
            for y in range(8, 22):
                img.putpixel((x, y), (220, 40, 40))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        with grpc.insecure_channel(f"127.0.0.1:{port}") as ch:
            stub = pb_grpc.ReconstructionServiceStub(ch)
            seg = stub.Segment(pb.SegmentRequest(client_id="t", image_data=buf.getvalue(),
                                                 points=[pb.SegPoint(x=15, y=12)]), timeout=10)
            assert seg.success and seg.mask_width == 40 and seg.mask_height == 30
            rec = stub.Reconstruct(pb.ReconstructRequest(client_id="t", image_data=buf.getvalue(),
                                                         mask_data=seg.mask_data, mask_width=40, mask_height=30),
                                   timeout=10)
            assert rec.success and rec.glb_file[:4] == b"glTF" and b"obj_2_pt3d_transform" in rec.json_file
    finally:
        proc.terminate()
        code = proc.wait(timeout=15)
        # Windows has no SIGTERM: terminate() is TerminateProcess, which always exits 1.
        assert code == 0 or sys.platform == "win32", "the stub must exit 0 on SIGTERM"


def test_inference_is_built_with_or_without_a_device_argument():
    from sam3d_scan_server.server import build_inference

    class New:
        def __init__(self, config_file, device=None, compile=False):
            self.args = (config_file, device, compile)

    class Pinned:
        def __init__(self, config_file, compile=False):
            self.args = (config_file, compile)

    assert build_inference(New, "c.yaml", "cuda:0").args == ("c.yaml", "cuda:0", False)
    assert build_inference(Pinned, "c.yaml", "cuda:0").args == ("c.yaml", False)
