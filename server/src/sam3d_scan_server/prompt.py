# Copyright (c) 2026 Seoul National University
# SPDX-License-Identifier: MIT

"""Segment prompts: SegPoint.z as a label, so a client can send negative clicks and a box without a proto change.

Pure Python (no torch, no grpc), shared by the real server (server.py) and the CPU stub (stub.py).
"""
from __future__ import annotations

import math
from typing import List, Optional

# SegPoint.z carries the prompt label (the proto has no label or box field; z was unused and is sent end to end by
# the native client). Legacy clients send z = 0 for every click, so 0 must keep meaning "positive click".
#   z = 0 or 1  positive click        z = -1  negative click (leave this out)
#   z = 2       box top-left corner   z = 3   box bottom-right corner (SAM's own corner codes)
LABEL_POSITIVE_CODES = (0, 1)
LABEL_NEGATIVE = -1
LABEL_BOX_TOP_LEFT = 2
LABEL_BOX_BOTTOM_RIGHT = 3
MIN_BOX_PX = 2.0


class PromptError(ValueError):
    """A Segment request whose prompt cannot be used (the request fails with this message)."""


class Prompt:
    """What one Segment request asks SAM3 for: clicks with labels (1 / 0) and at most one box (x1, y1, x2, y2)."""

    def __init__(self, points: List[List[float]], labels: List[int], box: Optional[List[float]]):
        self.points, self.labels, self.box = points, labels, box

    def summary(self) -> str:
        """The marker clients read back: 'box:1' tells a client that this server used its box."""
        pos = sum(1 for v in self.labels if v == 1)
        return f"points:{len(self.points)} | pos:{pos} | neg:{len(self.points) - pos} | box:{1 if self.box else 0}"


def _code(z: float) -> int:
    if not math.isfinite(z):
        raise PromptError(f"point label z={z} is not a finite number")
    code = int(round(z))
    if abs(z - code) > 1e-3:
        raise PromptError(f"point label z={z} is not an integer code")
    return code


def parse_prompt(points, width: int, height: int) -> Prompt:
    """Reads the labels out of SegPoint.z. Clicks outside the image are dropped (as before); the box is clamped to the
    image. A request needs a box or at least one positive click; negative clicks alone select nothing."""
    clicks, labels = [], []
    top_left, bottom_right = [], []
    for p in points:
        code = _code(p.z)
        if code in LABEL_POSITIVE_CODES or code == LABEL_NEGATIVE:
            if 0 <= p.x < width and 0 <= p.y < height:
                clicks.append([p.x, p.y])
                labels.append(0 if code == LABEL_NEGATIVE else 1)
        elif code == LABEL_BOX_TOP_LEFT:
            top_left.append(p)
        elif code == LABEL_BOX_BOTTOM_RIGHT:
            bottom_right.append(p)
        else:
            raise PromptError(f"unknown point label z={code} (0/1 positive, -1 negative, 2/3 box corners)")
    box = None
    if top_left or bottom_right:
        if len(top_left) != 1 or len(bottom_right) != 1:
            raise PromptError(f"a box needs exactly one z=2 and one z=3 corner (got {len(top_left)} and {len(bottom_right)})")
        a, b = top_left[0], bottom_right[0]
        # Box coordinates are continuous (pixel edges), so the far edge may be the image's width / height.
        x1, x2 = sorted((min(max(a.x, 0.0), float(width)), min(max(b.x, 0.0), float(width))))
        y1, y2 = sorted((min(max(a.y, 0.0), float(height)), min(max(b.y, 0.0), float(height))))
        if x2 - x1 < MIN_BOX_PX or y2 - y1 < MIN_BOX_PX:
            raise PromptError(f"the box is smaller than {MIN_BOX_PX:.0f} px inside the {width}x{height} image after clamping")
        box = [float(x1), float(y1), float(x2), float(y2)]
    if box is None and not any(v == 1 for v in labels):
        if labels:
            raise PromptError("only negative points: add a positive point or a box")
        raise PromptError("No valid points found within image bounds.")
    return Prompt(clicks, labels, box)


def single_mask(prompt: Prompt) -> bool:
    """SAM's guidance: ask for one mask (multimask_output=False) when the prompt is unambiguous, i.e. a box or two or
    more clicks; a single click keeps the three candidates and the best-IoU pick (the server's old behaviour)."""
    return prompt.box is not None or len(prompt.points) >= 2


def sam3_prompt_kwargs(prompt: Prompt) -> dict:
    """The Sam3TrackerProcessor arguments for one image and one object: points [image][object][point][xy], labels
    [image][object][point], boxes [image][box][xyxy]. Points and box are passed together when both are present."""
    kwargs = {}
    if prompt.points:
        kwargs["input_points"] = [[prompt.points]]
        kwargs["input_labels"] = [[prompt.labels]]
    if prompt.box is not None:
        kwargs["input_boxes"] = [[prompt.box]]
    return kwargs
