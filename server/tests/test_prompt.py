# Copyright (c) 2026 Seoul National University
# SPDX-License-Identifier: MIT

"""SegPoint.z as the prompt label: legacy clicks, box-only, box + negative clicks, and the errors. CPU only."""
import io
import os
import queue
import socket
import subprocess
import sys

import grpc
import numpy as np
import pytest
from PIL import Image

from sam3d_scan_server import protocol
from sam3d_scan_server.prompt import PromptError, parse_prompt, sam3_prompt_kwargs

pb, pb_grpc = protocol.load()


def pt(x, y, z=0.0):
    return pb.SegPoint(x=x, y=y, z=z)


def test_legacy_clicks_with_z0_stay_positive_and_out_of_bounds_are_dropped():
    p = parse_prompt([pt(1, 1), pt(50, 1), pt(3, 4)], 10, 10)
    assert p.points == [[1.0, 1.0], [3.0, 4.0]] and p.labels == [1, 1] and p.box is None
    assert sam3_prompt_kwargs(p) == {"input_points": [[[[1.0, 1.0], [3.0, 4.0]]]], "input_labels": [[[1, 1]]]}
    assert "box:0" in p.summary()


def test_z1_is_positive_too():
    assert parse_prompt([pt(2, 2, 1)], 10, 10).labels == [1]


def test_box_only():
    p = parse_prompt([pt(2, 3, 2), pt(8, 9, 3)], 10, 10)
    assert p.box == [2.0, 3.0, 8.0, 9.0] and p.points == [] and p.labels == []
    assert sam3_prompt_kwargs(p) == {"input_boxes": [[[2.0, 3.0, 8.0, 9.0]]]}
    assert "box:1" in p.summary()


def test_box_with_negative_click():
    p = parse_prompt([pt(1, 1, 2), pt(9, 9, 3), pt(5, 5, -1)], 10, 10)
    assert p.box == [1.0, 1.0, 9.0, 9.0] and p.points == [[5.0, 5.0]] and p.labels == [0]
    kw = sam3_prompt_kwargs(p)
    assert kw["input_boxes"] == [[[1.0, 1.0, 9.0, 9.0]]]
    assert kw["input_points"] == [[[[5.0, 5.0]]]] and kw["input_labels"] == [[[0]]]
    assert "neg:1" in p.summary() and "box:1" in p.summary()


def test_box_corners_are_ordered_and_clamped():
    p = parse_prompt([pt(120, -5, 2), pt(-3, 40, 3)], 100, 50)
    assert p.box == [0.0, 0.0, 100.0, 40.0]


def test_single_mask_for_boxes_and_multi_click_prompts():
    from sam3d_scan_server.prompt import single_mask
    assert not single_mask(parse_prompt([pt(1, 1)], 10, 10))
    assert single_mask(parse_prompt([pt(1, 1), pt(2, 2)], 10, 10))
    assert single_mask(parse_prompt([pt(1, 1, 2), pt(8, 8, 3)], 10, 10))


@pytest.mark.parametrize("points, why", [
    ([pt(1, 1, 2)], "exactly one"),
    ([pt(1, 1, 2), pt(5, 5, 3), pt(6, 6, 3)], "exactly one"),
    ([pt(1, 1, 7)], "unknown point label"),
    ([pt(1, 1, 0.5)], "not an integer"),
    ([pt(1, 1, -1)], "only negative"),
    ([pt(50, 50)], "No valid points"),
    ([pt(200, 200, 2), pt(300, 300, 3)], "smaller than"),
    ([pt(1, 1, float("nan"))], "not a finite"),
])
def test_unusable_prompts_fail_with_a_reason(points, why):
    with pytest.raises(PromptError, match=why):
        parse_prompt(points, 10, 10)


def png(w, h, fill, rect=None, rect_color=(220, 40, 40)):
    img = Image.new("RGB", (w, h), fill)
    if rect:
        x1, y1, x2, y2 = rect
        for x in range(x1, x2):
            for y in range(y1, y2):
                img.putpixel((x, y), rect_color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_worker_reports_the_box_marker_and_fails_bad_prompts():
    """The real worker's request loop with the model replaced: the parsed prompt reaches the model and the reply
    says box:1, which is what the Unity client reads to know this server honours boxes."""
    from sam3d_scan_server.config import ServerConfig
    from sam3d_scan_server.server import Sam3Worker

    seen = []

    class W(Sam3Worker):
        def _load_models(self):
            self.ready.set()

        def _run_sam3_inference(self, image, prompt):
            seen.append(sam3_prompt_kwargs(prompt))
            return np.ones((image.size[1], image.size[0]), dtype=np.uint8)

    q, responses = queue.Queue(), {"a": queue.Queue(), "b": queue.Queue()}
    w = W(0, q, responses, ServerConfig(gpus=[0]))
    w.start()
    q.put(("a", "c", png(10, 8, (0, 0, 0)), [pt(1, 1, 2), pt(6, 5, 3), pt(3, 3, -1)]))
    ok = responses["a"].get(timeout=5)
    assert ok["success"] and "box:1" in ok["message"] and "neg:1" in ok["message"]
    assert seen[0]["input_boxes"] == [[[1.0, 1.0, 6.0, 5.0]]] and seen[0]["input_labels"] == [[[0]]]
    q.put(("b", "c", png(10, 8, (0, 0, 0)), [pt(1, 1, 2)]))
    bad = responses["b"].get(timeout=5)
    assert not bad["success"] and "exactly one" in bad["message"]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_stub_honours_box_and_legacy_points():
    from sam3d_scan_server.cli import check_health
    port = free_port()
    proc = subprocess.Popen([sys.executable, "-m", "sam3d_scan_server", "--stub", "--host", "127.0.0.1",
                             "--port", str(port)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        assert check_health(f"127.0.0.1:{port}", timeout=2.0, wait=30.0) == 0
        # Two red rectangles; the box covers only the left one.
        img = Image.new("RGB", (40, 30), (10, 10, 10))
        for x0 in (4, 24):
            for x in range(x0, x0 + 10):
                for y in range(8, 22):
                    img.putpixel((x, y), (220, 40, 40))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        with grpc.insecure_channel(f"127.0.0.1:{port}") as ch:
            stub = pb_grpc.ReconstructionServiceStub(ch)
            boxed = stub.Segment(pb.SegmentRequest(client_id="t", image_data=buf.getvalue(),
                                                   points=[pt(2, 6, 2), pt(16, 24, 3)]), timeout=10)
            assert boxed.success and "box:1" in boxed.message
            m = np.frombuffer(boxed.mask_data, dtype=np.uint8).reshape(30, 40)
            assert m[15, 8] == 1 and m[15, 28] == 0
            legacy = stub.Segment(pb.SegmentRequest(client_id="t", image_data=buf.getvalue(),
                                                    points=[pt(8, 15)]), timeout=10)
            assert legacy.success and "box:0" in legacy.message
            m = np.frombuffer(legacy.mask_data, dtype=np.uint8).reshape(30, 40)
            assert m[15, 8] == 1 and m[15, 28] == 1
    finally:
        proc.terminate()
        proc.wait(timeout=15)


REAL = os.environ.get("SAM3D_SCAN_REAL_SERVER")   # host:port of a running real server (plaintext, e.g. on the host)


@pytest.mark.skipif(not REAL, reason="set SAM3D_SCAN_REAL_SERVER=host:port to run against the real model")
@pytest.mark.parametrize("extra", [[], [(330.0, 260.0, -1.0)]], ids=["box-only", "box+negative"])
def test_real_model_honours_the_box(extra):
    """The real SAM3 with a box (and a negative click): box:1 in the reply and a mask that lies mostly in the box.
    Image: two red squares on grey; the box covers the left one only."""
    img = Image.new("RGB", (640, 480), (128, 128, 128))
    for x0 in (120, 380):
        for x in range(x0, x0 + 140):
            for y in range(170, 310):
                img.putpixel((x, y), (210, 40, 40))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    with grpc.insecure_channel(REAL) as ch:
        stub = pb_grpc.ReconstructionServiceStub(ch)
        points = [pt(100, 150, 2), pt(280, 330, 3)] + [pt(*e) for e in extra]
        r = stub.Segment(pb.SegmentRequest(client_id="test", image_data=buf.getvalue(), points=points), timeout=120)
    assert r.success and "box:1" in r.message, r.message
    m = np.frombuffer(r.mask_data, dtype=np.uint8).reshape(r.mask_height, r.mask_width)
    inside = m[150:330, 100:280].sum()
    assert m.sum() > 0 and inside / m.sum() > 0.9


def test_model_loads_of_the_two_pools_never_overlap():
    """Regression for the torchvision import race between the two pools: both workers load under one lock."""
    import threading
    import time
    from sam3d_scan_server.config import ServerConfig
    from sam3d_scan_server.server import Sam3dObjectWorker, Sam3Worker

    active, overlap = [0], [False]
    guard = threading.Lock()

    def fake_load(self):
        with guard:
            active[0] += 1
            overlap[0] |= active[0] > 1
        time.sleep(0.2)
        with guard:
            active[0] -= 1
        self.ready.set()

    class A(Sam3Worker):
        _load_models_locked = fake_load

    class B(Sam3dObjectWorker):
        _load_models_locked = fake_load

    cfg = ServerConfig(gpus=[0])
    workers = [A(0, queue.Queue(), {}, cfg), B(0, queue.Queue(), {}, cfg)]
    threads = [threading.Thread(target=w._load_models) for w in workers]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert all(w.ready.is_set() for w in workers) and not overlap[0]
