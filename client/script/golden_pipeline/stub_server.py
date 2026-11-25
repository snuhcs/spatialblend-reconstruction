# Copyright (c) 2026 Seoul National University
# SPDX-License-Identifier: MIT

"""Deterministic stand-in for the scan server (../../../server), used ONLY to prove the golden
harness (library -> wire -> files -> metrics) works where no SAM3 / SAM 3D checkpoints exist.

It speaks distributed.proto and writes the same file schema as the real server
(format_transform_to_json). It does not run any model: the mask is the colour region at the
first click, the mesh is an ellipsoid sized from the mask. Goldens recorded against it are
meaningless as model goldens; the harness marks them `server_kind: stub`.

    python stub_server.py --port 9810
"""
from __future__ import annotations

import argparse
import io
import json
import struct
import sys
import tempfile
import time
from concurrent import futures
from pathlib import Path

import numpy as np
from PIL import Image

import grpc

HERE = Path(__file__).resolve().parent


# The repository's top-level proto/, shared with the server (client/script/golden_pipeline -> repository root).
PROTO_DIR = HERE.parents[2] / "proto"


def _load_pb():
    """Generates the Python stubs from the shared distributed.proto into a temp dir (grpcio-tools)."""
    from grpc_tools import protoc
    out = tempfile.mkdtemp(prefix="dist_pb_")
    rc = protoc.main(["protoc", f"-I{PROTO_DIR}", f"--python_out={out}", f"--grpc_python_out={out}",
                      str(PROTO_DIR / "distributed.proto")])
    if rc != 0:
        raise RuntimeError("protoc failed")
    sys.path.insert(0, out)
    import distributed_pb2, distributed_pb2_grpc  # noqa: E401
    return distributed_pb2, distributed_pb2_grpc


pb, pb_grpc = _load_pb()


def ellipsoid_glb(rx: float, ry: float, rz: float, nu: int = 24, nv: int = 16) -> bytes:
    verts = []
    for i in range(nv + 1):
        th = np.pi * i / nv
        for j in range(nu):
            ph = 2 * np.pi * j / nu
            verts.append((rx * np.sin(th) * np.cos(ph), ry * np.cos(th), rz * np.sin(th) * np.sin(ph)))
    idx = []
    for i in range(nv):
        for j in range(nu):
            a, b = i * nu + j, i * nu + (j + 1) % nu
            c, d = (i + 1) * nu + j, (i + 1) * nu + (j + 1) % nu
            idx += [a, c, b, b, c, d]
    v = np.asarray(verts, dtype="<f4")
    ix = np.asarray(idx, dtype="<u4")
    vb, ib = v.tobytes(), ix.tobytes()
    blob = vb + ib
    doc = {
        "asset": {"version": "2.0", "generator": "golden-stub"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(vb), "target": 34962},
                        {"buffer": 0, "byteOffset": len(vb), "byteLength": len(ib), "target": 34963}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": len(v), "type": "VEC3",
                       "min": v.min(0).tolist(), "max": v.max(0).tolist()},
                      {"bufferView": 1, "componentType": 5125, "count": len(ix), "type": "SCALAR"}],
    }
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    blob += b"\0" * (-len(blob) % 4)
    total = 12 + 8 + len(js) + 8 + len(blob)
    return (b"glTF" + struct.pack("<II", 2, total) + struct.pack("<II", len(js), 0x4E4F534A) + js
            + struct.pack("<II", len(blob), 0x004E4942) + blob)


class Servicer(pb_grpc.ReconstructionServiceServicer):
    def Segment(self, request, context):
        img = np.asarray(Image.open(io.BytesIO(request.image_data)).convert("RGB")).astype(np.int32)
        if not request.points:
            return pb.SegmentResponse(success=False, message="no points")
        p = request.points[0]
        x, y = int(round(p.x)), int(round(p.y))
        h, w, _ = img.shape
        if not (0 <= x < w and 0 <= y < h):
            return pb.SegmentResponse(success=False, message="point outside image")
        mask = (np.abs(img - img[y, x]).sum(-1) < 60).astype(np.uint8)
        return pb.SegmentResponse(success=True, message="ok", mask_data=mask.tobytes(),
                                  mask_width=w, mask_height=h)

    def Reconstruct(self, request, context):
        w, h = request.mask_width, request.mask_height
        if len(request.mask_data) != w * h:
            return pb.ReconstructResponse(success=False, message="mask size mismatch")
        m = np.frombuffer(request.mask_data, dtype=np.uint8).reshape(h, w)
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            return pb.ReconstructResponse(success=False, message="empty mask")
        bw, bh = (xs.max() - xs.min() + 1) / w, (ys.max() - ys.min() + 1) / h
        glb = ellipsoid_glb(bw, bh, 0.5 * min(bw, bh))
        scale = 1.25
        q = np.array([0.4268, 0.6793, 0.5297, 0.2752])
        q /= np.linalg.norm(q)
        qw, qx, qy, qz = q[3], q[0], q[1], q[2]  # same ordering the stored server result uses (xyzw)
        R = np.array([[1 - 2 * (qy**2 + qz**2), 2 * (qx*qy - qz*qw), 2 * (qx*qz + qy*qw)],
                      [2 * (qx*qy + qz*qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy*qz - qx*qw)],
                      [2 * (qx*qz - qy*qw), 2 * (qy*qz + qx*qw), 1 - 2 * (qx**2 + qy**2)]])
        t = np.array([(xs.mean() / w - 0.5), (ys.mean() / h - 0.5), 1.5])
        M = np.eye(4)
        M[:3, :3] = R * scale
        M[:3, 3] = t
        info = {"filename": "reconstruction", "timestamp": "stub",
                "obj_2_pt3d_transform": {"scale": [scale] * 3, "rotationQuaternion": q.tolist(),
                                         "translation": t.tolist(), "matrix": M.flatten().tolist()}}
        return pb.ReconstructResponse(success=True, message="ok", glb_file=glb,
                                      json_file=json.dumps(info).encode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    opts = [("grpc.max_receive_message_length", 100 * 1024 * 1024),
            ("grpc.max_send_message_length", 100 * 1024 * 1024)]
    srv = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=opts)
    pb_grpc.add_ReconstructionServiceServicer_to_server(Servicer(), srv)
    if srv.add_insecure_port(f"{a.host}:{a.port}") == 0:
        sys.exit(f"cannot bind {a.host}:{a.port}")
    srv.start()
    print(f"stub listening on {a.host}:{a.port}", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        srv.stop(0)


if __name__ == "__main__":
    main()
