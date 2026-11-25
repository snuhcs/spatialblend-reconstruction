# Copyright (c) 2026 Seoul National University
# SPDX-License-Identifier: MIT

"""Deterministic stand-in for the scan server (server.py), used ONLY to prove the client pipeline
(library -> wire -> files -> metrics) works where no SAM3 / SAM 3D checkpoints exist.

It speaks distributed.proto and writes the same file schema as the real server
(format_transform_to_json). It does not run any model: the mask is the colour region at the
first click, the mesh is an ellipsoid sized from the mask. Goldens recorded against it are
meaningless as model goldens; the harness marks them `server_kind: stub`.

    python -m sam3d_scan_server.stub --port 9810        # or: sam3d-scan-server --stub --port 9810

Written by Jingyu Lee for the client's golden-pipeline tests. It answers the protocol that
Dongho Han designed for server_distributed.py.
"""
from __future__ import annotations

import argparse
import io
import json
import struct
import sys
import time
from concurrent import futures

import numpy as np
from PIL import Image

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from . import protocol

pb, pb_grpc = protocol.load()


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
        from .prompt import PromptError, parse_prompt
        img = np.asarray(Image.open(io.BytesIO(request.image_data)).convert("RGB")).astype(np.int32)
        if not request.points:
            return pb.SegmentResponse(success=False, message="no points")
        h, w, _ = img.shape
        try:
            prompt = parse_prompt(request.points, w, h)
        except PromptError as e:
            return pb.SegmentResponse(success=False, message=str(e))
        # The seed is the first positive click, else the box centre. With a box the mask is limited to the box.
        seeds = [pt for pt, lab in zip(prompt.points, prompt.labels) if lab == 1]
        if seeds:
            x, y = int(round(seeds[0][0])), int(round(seeds[0][1]))
        else:
            b = prompt.box
            x, y = min(int((b[0] + b[2]) / 2), w - 1), min(int((b[1] + b[3]) / 2), h - 1)
        mask = (np.abs(img - img[y, x]).sum(-1) < 60).astype(np.uint8)
        if prompt.box is not None:
            x1, y1, x2, y2 = (int(round(v)) for v in prompt.box)
            inside = np.zeros_like(mask)
            inside[y1:y2, x1:x2] = 1
            mask &= inside
        # Stand-in for a negative click: drop that click's colour (image-wide, unlike SAM; enough for path tests).
        for (nx, ny), lab in zip(prompt.points, prompt.labels):
            if lab == 0:
                mask[(np.abs(img - img[int(ny), int(nx)]).sum(-1) < 60)] = 0
        return pb.SegmentResponse(success=True, message="ok | " + prompt.summary(), mask_data=mask.tobytes(),
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
    hs = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(hs, srv)
    for name in ("", "spatialblend.scan.v1.ReconstructionService"):
        hs.set(name, health_pb2.HealthCheckResponse.SERVING)
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
