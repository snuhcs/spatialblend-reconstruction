#!/usr/bin/env python3
# Copyright (c) 2026 Seoul National University
# SPDX-License-Identifier: MIT

"""Writes the synthetic test images and click files under session/inference/sam3d/test_data/.

The images are drawn here (a red block and a blue disc on grey), so they carry no third-party or personal content
(CC0). The remote tests (sam3d_remote_test, distributed_test; both tagged manual, they need a running scan server) pair
each image with the click file of the same stem: one "x, y" per line in image pixels.

    python3 script/make_test_data.py        # needs Pillow
"""
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parents[1] / "session" / "inference" / "sam3d" / "test_data"


def main() -> None:
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "points").mkdir(parents=True, exist_ok=True)

    block = Image.new("RGB", (640, 480), (128, 128, 128))
    d = ImageDraw.Draw(block)
    d.rectangle([220, 140, 420, 340], fill=(200, 40, 40))
    d.rectangle([220, 330, 420, 360], fill=(90, 90, 90))  # a shadow, so the object is not the only edge
    block.save(OUT / "images" / "synthetic_block.png", optimize=True)
    (OUT / "points" / "synthetic_block.txt").write_text("320, 240\n260, 180\n380, 300\n")

    disc = Image.new("RGB", (640, 480), (150, 150, 140))
    d = ImageDraw.Draw(disc)
    d.ellipse([200, 120, 440, 360], fill=(40, 70, 200))
    disc.save(OUT / "images" / "synthetic_disc.png", optimize=True)
    (OUT / "points" / "synthetic_disc.txt").write_text("320, 240\n")


if __name__ == "__main__":
    main()
