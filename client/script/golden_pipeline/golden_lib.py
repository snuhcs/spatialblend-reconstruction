# Copyright (c) 2026 Seoul National University
# SPDX-License-Identifier: MIT

"""Shared pieces of the SAM3 / SAM3D golden pipeline: the fixed synthetic corpus,
a minimal GLB reader, the metrics, and the drift comparison.

Nothing here talks to a server. Inputs are generated from a seed, so no dataset
image is committed and the corpus is identical on every host.
"""
from __future__ import annotations

import io
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------

IMG_W, IMG_H = 512, 384


@dataclass(frozen=True)
class Case:
    name: str
    shape: str  # disk | box | ellipse
    color: tuple
    center: tuple
    size: tuple  # disk: (r, r); box/ellipse: (half_w, half_h)
    points: tuple  # click points (x, y), top-left origin, inside the shape


CASES = (
    Case("disk", "disk", (200, 40, 40), (256, 190), (90, 90), ((256, 190), (280, 170))),
    Case("box", "box", (40, 60, 210), (230, 200), (110, 70), ((230, 200), (180, 220))),
    Case("ellipse", "ellipse", (30, 160, 70), (300, 180), (120, 60), ((300, 180), (340, 190))),
)


def shape_mask(case: Case) -> np.ndarray:
    """Analytic ground-truth mask (uint8 0/1, shape [H, W])."""
    yy, xx = np.mgrid[0:IMG_H, 0:IMG_W]
    cx, cy = case.center
    a, b = case.size
    if case.shape == "box":
        m = (np.abs(xx - cx) <= a) & (np.abs(yy - cy) <= b)
    else:  # disk and ellipse
        m = ((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2 <= 1.0
    return m.astype(np.uint8)


def render_image(case: Case, seed: int = 7) -> bytes:
    """Deterministic PNG bytes: noisy grey gradient background, one flat-colour shape."""
    rng = np.random.default_rng(seed + sum(map(ord, case.name)))
    yy, xx = np.mgrid[0:IMG_H, 0:IMG_W]
    base = 110 + 50 * (xx / IMG_W) + 30 * (yy / IMG_H)
    img = np.stack([base, base, base], axis=-1) + rng.normal(0, 3, (IMG_H, IMG_W, 3))
    img = np.clip(img, 0, 255).astype(np.uint8)
    img[shape_mask(case) == 1] = case.color
    buf = io.BytesIO()
    Image.fromarray(img, "RGB").save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


# --------------------------------------------------------------------------
# GLB reader (positions + index count; one scene, node transforms ignored)
# --------------------------------------------------------------------------

_COMPONENT = {5120: "b", 5121: "B", 5122: "h", 5123: "H", 5125: "I", 5126: "f"}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def read_glb(path: Path):
    """Returns (positions float32 [N,3], n_indices, doc). Raises ValueError on a bad file."""
    data = Path(path).read_bytes()
    if len(data) < 20 or data[:4] != b"glTF":
        raise ValueError("not a GLB (bad magic)")
    version, total = struct.unpack_from("<II", data, 4)
    if version != 2 or total != len(data):
        raise ValueError(f"bad GLB header (version={version}, length={total}, file={len(data)})")
    off, doc, binblob = 12, None, b""
    while off + 8 <= len(data):
        clen, ctype = struct.unpack_from("<II", data, off)
        chunk = data[off + 8: off + 8 + clen]
        if ctype == 0x4E4F534A:
            doc = json.loads(chunk.decode("utf-8"))
        elif ctype == 0x004E4942 and not binblob:
            binblob = chunk
        off += 8 + clen
    if doc is None:
        raise ValueError("GLB has no JSON chunk")

    def accessor(i):
        acc = doc["accessors"][i]
        bv = doc["bufferViews"][acc["bufferView"]]
        fmt = _COMPONENT[acc["componentType"]]
        n, nc = acc["count"], _NCOMP[acc["type"]]
        size = struct.calcsize(fmt)
        stride = bv.get("byteStride") or size * nc
        base = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
        out = np.empty((n, nc), dtype=np.dtype(fmt).newbyteorder("<"))
        for k in range(nc):
            out[:, k] = np.ndarray((n,), dtype=out.dtype, buffer=binblob, offset=base + k * size,
                                   strides=(stride,))
        return out

    pos, n_idx = [], 0
    for mesh in doc.get("meshes", []):
        for prim in mesh.get("primitives", []):
            pos.append(accessor(prim["attributes"]["POSITION"]).astype(np.float32))
            if "indices" in prim:
                n_idx += doc["accessors"][prim["indices"]]["count"]
            else:
                n_idx += len(pos[-1])
    if not pos:
        raise ValueError("GLB has no mesh primitive")
    return np.concatenate(pos), n_idx, doc


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    inter = int(np.logical_and(a > 0, b > 0).sum())
    union = int(np.logical_or(a > 0, b > 0).sum())
    return 1.0 if union == 0 else inter / union


def _normalise(p: np.ndarray) -> np.ndarray:
    lo, hi = p.min(0), p.max(0)
    return (p - (lo + hi) / 2) / max(float((hi - lo).max()), 1e-12)


def _sample(p: np.ndarray, n: int) -> np.ndarray:
    if len(p) <= n:
        return p
    return p[np.linspace(0, len(p) - 1, n).astype(np.int64)]  # deterministic stride


def chamfer(a: np.ndarray, b: np.ndarray, n: int = 1024) -> float:
    """Symmetric mean nearest-neighbour distance after centring and scaling each to longest side 1."""
    a, b = _sample(_normalise(a.astype(np.float64)), n), _sample(_normalise(b.astype(np.float64)), n)
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    return float(d.min(1).mean() + d.min(0).mean())


def mesh_metrics(glb_path: Path) -> dict:
    pos, n_idx, doc = read_glb(glb_path)
    lo, hi = pos.min(0), pos.max(0)
    ext = (hi - lo)
    longest = float(ext.max())
    return {
        "bytes": Path(glb_path).stat().st_size,
        "vertex_count": int(len(pos)),
        "face_count": int(n_idx // 3),
        "bounds_min": [round(float(x), 6) for x in lo],
        "bounds_max": [round(float(x), 6) for x in hi],
        "extent": [round(float(x), 6) for x in ext],
        "scale": round(longest, 6),  # longest side in model units
        "aspect": [round(float(x) / longest, 6) for x in ext] if longest > 0 else [0, 0, 0],
        "has_materials": bool(doc.get("materials")),
        "finite": bool(np.isfinite(pos).all()),
    }


OBJINFO_KEYS = {"filename", "timestamp", "obj_2_pt3d_transform"}
TRANSFORM_KEYS = {"scale", "rotationQuaternion", "translation", "matrix"}


def objinfo_problems(info: dict) -> list:
    """Schema and consistency checks of {prefix}_objinfo.json against what the server writes and
    what ScanObjectFit (Unity) reads: obj_2_pt3d_transform.matrix, 16 floats, row-major, column-vector
    convention (translation in matrix[3], [7], [11], bottom row 0 0 0 1)."""
    p = []
    if not OBJINFO_KEYS <= set(info):
        p.append(f"missing top-level keys {sorted(OBJINFO_KEYS - set(info))}")
        return p
    t = info["obj_2_pt3d_transform"]
    if not TRANSFORM_KEYS <= set(t):
        p.append(f"missing transform keys {sorted(TRANSFORM_KEYS - set(t))}")
        return p
    m = t["matrix"]
    if len(m) != 16 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in m):
        p.append("matrix is not 16 finite numbers")
        return p
    M = np.array(m, dtype=np.float64).reshape(4, 4)
    if len(t["scale"]) != 3 or len(t["rotationQuaternion"]) != 4 or len(t["translation"]) != 3:
        p.append("scale/rotationQuaternion/translation have the wrong length")
        return p
    if not np.allclose(M[3], [0, 0, 0, 1], atol=1e-5):
        p.append(f"bottom row is {M[3].tolist()}, not 0 0 0 1")
    if not np.allclose(M[:3, 3], t["translation"], atol=1e-4):
        p.append("matrix translation column differs from 'translation'")
    col = np.linalg.norm(M[:3, :3], axis=0)
    if not np.allclose(col, t["scale"], rtol=1e-3, atol=1e-4):
        p.append(f"matrix column norms {col.round(5).tolist()} differ from 'scale' {t['scale']}")
    if np.linalg.det(M[:3, :3]) <= 0:
        p.append("matrix has a non-positive determinant (mirrored)")
    if abs(np.linalg.norm(t["rotationQuaternion"]) - 1) > 1e-3:
        p.append("rotationQuaternion is not unit length")
    return p


def unity_object_to_world(info: dict, cam_c2w: np.ndarray) -> np.ndarray:
    """ScanObjectFit.ObjectToWorld: cam2world * rhs2lhs * obj2pt3d * lhs2rhs (X flip both sides)."""
    flip = np.diag([-1.0, 1.0, 1.0, 1.0])
    obj = np.array(info["obj_2_pt3d_transform"]["matrix"], dtype=np.float64).reshape(4, 4)
    return cam_c2w @ flip.T @ obj @ flip


def quat_angle_deg(q1, q2) -> float:
    d = abs(float(np.dot(q1, q2)) / (np.linalg.norm(q1) * np.linalg.norm(q2)))
    return math.degrees(2 * math.acos(min(1.0, d)))


# --------------------------------------------------------------------------
# Drift comparison
# --------------------------------------------------------------------------

def compare(case: str, fresh: dict, golden: dict, thr: dict) -> list:
    """fresh/golden = {'mask': ndarray, 'glb': Path, 'objinfo': dict}. Returns (name, value, limit, ok) rows."""
    rows = []

    def add(name, value, limit, ok):
        rows.append({"case": case, "metric": name, "value": value, "limit": limit, "ok": bool(ok)})

    iou = mask_iou(fresh["mask"], golden["mask"])
    add("mask_iou", round(iou, 4), f">= {thr['mask_iou_min']}", iou >= thr["mask_iou_min"])

    fm, gm = mesh_metrics(fresh["glb"]), mesh_metrics(golden["glb"])
    band = thr["vertex_count_rel"]
    rel = abs(fm["vertex_count"] - gm["vertex_count"]) / max(gm["vertex_count"], 1)
    add("vertex_count_rel_diff", round(rel, 4), f"<= {band}", rel <= band)
    srel = abs(fm["scale"] - gm["scale"]) / max(gm["scale"], 1e-12)
    add("mesh_scale_rel_diff", round(srel, 4), f"<= {thr['mesh_scale_rel']}", srel <= thr["mesh_scale_rel"])
    adiff = float(np.abs(np.array(fm["aspect"]) - np.array(gm["aspect"])).max())
    add("aspect_abs_diff", round(adiff, 4), f"<= {thr['aspect_abs']}", adiff <= thr["aspect_abs"])
    cd = chamfer(read_glb(fresh["glb"])[0], read_glb(golden["glb"])[0])
    add("chamfer_normalised", round(cd, 5), f"<= {thr['chamfer_max']}", cd <= thr["chamfer_max"])
    add("mesh_finite", fm["finite"], "true", fm["finite"])

    fo, go = fresh["objinfo"]["obj_2_pt3d_transform"], golden["objinfo"]["obj_2_pt3d_transform"]
    sd = float(np.abs(np.array(fo["scale"]) / np.array(go["scale"]) - 1).max())
    add("objinfo_scale_rel_diff", round(sd, 4), f"<= {thr['objinfo_scale_rel']}", sd <= thr["objinfo_scale_rel"])
    td = float(np.linalg.norm(np.array(fo["translation"]) - np.array(go["translation"])))
    add("objinfo_translation_diff", round(td, 4), f"<= {thr['objinfo_translation_abs']}",
        td <= thr["objinfo_translation_abs"])
    ra = quat_angle_deg(fo["rotationQuaternion"], go["rotationQuaternion"])
    add("objinfo_rotation_deg", round(ra, 3), f"<= {thr['objinfo_rotation_deg']}", ra <= thr["objinfo_rotation_deg"])
    return rows
