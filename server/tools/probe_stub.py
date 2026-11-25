# Copyright (c) 2026 Seoul National University
# SPDX-License-Identifier: MIT

"""One Segment (box prompt) then one Reconstruct against a running server, on a synthetic image. Exit 0 on success.

    python tools/probe_stub.py HOST:PORT

Used by CI against the stub container; it works against the real server too (the mask is then SAM 3's).
"""
import io
import sys

import grpc
from PIL import Image

from sam3d_scan_server import protocol


def main(address: str) -> int:
    pb, pb_grpc = protocol.load()
    img = Image.new("RGB", (64, 48), (20, 20, 20))
    for x in range(16, 48):
        for y in range(12, 36):
            img.putpixel((x, y), (220, 40, 40))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    with grpc.insecure_channel(address) as ch:
        stub = pb_grpc.ReconstructionServiceStub(ch)
        box = [pb.SegPoint(x=8, y=6, z=2), pb.SegPoint(x=56, y=42, z=3)]
        seg = stub.Segment(pb.SegmentRequest(client_id="probe", image_data=buf.getvalue(), points=box), timeout=60)
        print("Segment:", seg.success, seg.message)
        if not (seg.success and "box:1" in seg.message and len(seg.mask_data) == seg.mask_width * seg.mask_height):
            return 1
        rec = stub.Reconstruct(pb.ReconstructRequest(client_id="probe", image_data=buf.getvalue(),
                                                     mask_data=seg.mask_data, mask_width=seg.mask_width,
                                                     mask_height=seg.mask_height), timeout=600)
        print("Reconstruct:", rec.success, rec.message, f"glb:{len(rec.glb_file)}B json:{len(rec.json_file)}B")
        return 0 if rec.success and rec.glb_file[:4] == b"glTF" else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
