#!/usr/bin/env python3
# Copyright (c) 2026 Seoul National University
# SPDX-License-Identifier: MIT

"""SAM3 / SAM3D golden pipeline: drives the real libspatialblend_session (ctypes) through
spatialblend_inference_segment and spatialblend_inference_reconstruct against a live scan server,
then compares the produced files with a stored golden and exits non-zero on drift.

  check   run the corpus, compare with --golden-dir          exit 0 ok, 1 drift, 2 setup error
  record  run the corpus N times, write goldens + calibration.json (run-to-run spread)
  selftest  start the bundled stub server, record then check against it (harness proof only)

The library is the exact binary the APK ships (build: script/build_c_api.py or
`bazel build //session/c:session_c`). Goldens are model outputs and live OUTSIDE the repo
(--golden-dir, a directory you keep); only thresholds.json is committed.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import golden_lib as G  # noqa: E402

TYPE_SEG, TYPE_OBJ = 2, 3
EXIT_OK, EXIT_DRIFT, EXIT_SETUP = 0, 1, 2

# A fixed camera-to-world pose (quarter turn about Y, translation) used to exercise the Unity composition.
CAM_C2W = np.array([[0, 0, 1, 1], [0, 1, 0, 2], [-1, 0, 0, 3], [0, 0, 0, 1]], dtype=np.float64)


class Point3(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double), ("z", ctypes.c_double)]


CB = ctypes.CFUNCTYPE(None, ctypes.c_int64, ctypes.c_int64, ctypes.c_bool, ctypes.c_char_p)


class Setup(Exception):
    pass


class Lib:
    def __init__(self, path: str):
        try:
            self.l = ctypes.CDLL(path)
        except OSError as e:
            raise Setup(f"cannot load {path}: {e}")
        l = self.l
        l.spatialblend_inference_session_create.restype = ctypes.c_int64
        l.spatialblend_inference_session_create.argtypes = [ctypes.c_int]
        l.spatialblend_inference_session_connect.restype = ctypes.c_int64
        l.spatialblend_inference_session_connect.argtypes = [ctypes.c_int64, ctypes.c_char_p]
        l.spatialblend_inference_session_destroy.restype = ctypes.c_int64
        l.spatialblend_inference_session_destroy.argtypes = [ctypes.c_int64]
        l.spatialblend_register_on_inference_complete.restype = ctypes.c_bool
        l.spatialblend_register_on_inference_complete.argtypes = [ctypes.c_int64, CB]
        l.spatialblend_try_wait_for_event.restype = ctypes.c_bool
        l.spatialblend_try_wait_for_event.argtypes = [ctypes.c_int64, ctypes.c_int64]
        l.spatialblend_get_error_message.restype = ctypes.c_char_p
        l.spatialblend_get_error_message.argtypes = [ctypes.c_int64]
        l.spatialblend_inference_segment.restype = ctypes.c_int64
        l.spatialblend_inference_segment.argtypes = [ctypes.c_int64, ctypes.c_char_p, ctypes.c_char_p,
                                                  ctypes.c_uint64, ctypes.POINTER(Point3), ctypes.c_uint64,
                                                  ctypes.c_char_p]
        l.spatialblend_inference_reconstruct.restype = ctypes.c_int64
        l.spatialblend_inference_reconstruct.argtypes = [ctypes.c_int64, ctypes.c_char_p, ctypes.c_char_p,
                                                      ctypes.c_uint64, ctypes.c_char_p, ctypes.c_uint64,
                                                      ctypes.c_int32, ctypes.c_int32, ctypes.c_char_p]
        self.done = {}
        self.cv = threading.Condition()

        def on_done(session, event, ok, msg):
            with self.cv:
                self.done[event] = (bool(ok), (msg or b"").decode(errors="replace"))
                self.cv.notify_all()

        self._cb = CB(on_done)  # keep alive

    def session(self, kind: int, addr: str, connect_timeout_s: float) -> int:
        s = self.l.spatialblend_inference_session_create(kind)
        if s == 0:
            raise Setup("session_create returned 0")
        self.l.spatialblend_register_on_inference_complete(s, self._cb)
        h = self.l.spatialblend_inference_session_connect(s, addr.encode())
        if h != 0 and not self.l.spatialblend_try_wait_for_event(h, int(connect_timeout_s * 1000)):
            msg = (self.l.spatialblend_get_error_message(h) or b"").decode(errors="replace")
            raise Setup(f"connect to {addr} failed: {msg or 'timeout'}")
        return s

    def await_event(self, handle: int, timeout_s: float):
        if handle == 0:
            return False, "call returned handle 0"
        end = time.monotonic() + timeout_s
        with self.cv:
            while handle not in self.done:
                left = end - time.monotonic()
                if left <= 0:
                    return False, f"no completion callback within {timeout_s}s"
                self.cv.wait(left)
            return self.done[handle]


def run_case(lib: Lib, seg: int, obj: int, case: G.Case, out: Path, seg_to: float, rec_to: float):
    """One image through the real segment and reconstruct calls. Returns (artifacts, latencies, problems)."""
    problems, lat = [], {}
    png = G.render_image(case)
    out.mkdir(parents=True, exist_ok=True)
    mask_path = out / "mask.u8"
    pts = (Point3 * len(case.points))(*[Point3(x, y, 0.0) for x, y in case.points])

    t = time.monotonic()
    ok, msg = lib.await_event(lib.l.spatialblend_inference_segment(
        seg, case.name.encode(), png, len(png), pts, len(case.points), str(mask_path).encode()), seg_to)
    lat["segment"] = time.monotonic() - t
    if not ok:
        return None, lat, [f"segment failed: {msg}"]
    if not mask_path.exists():
        return None, lat, ["segment reported success but wrote no mask file"]
    raw = mask_path.read_bytes()
    if len(raw) != G.IMG_W * G.IMG_H:
        return None, lat, [f"mask is {len(raw)} bytes, expected {G.IMG_W * G.IMG_H} (H*W uint8)"]
    mask = np.frombuffer(raw, dtype=np.uint8).reshape(G.IMG_H, G.IMG_W)
    if not set(np.unique(mask).tolist()) <= {0, 1}:
        problems.append("mask holds values other than 0/1")

    prefix = out / "recon"
    t = time.monotonic()
    ok, msg = lib.await_event(lib.l.spatialblend_inference_reconstruct(
        obj, case.name.encode(), png, len(png), raw, len(raw), G.IMG_W, G.IMG_H,
        str(prefix).encode()), rec_to)
    lat["reconstruct"] = time.monotonic() - t
    if not ok:
        return None, lat, problems + [f"reconstruct failed: {msg}"]

    glb, info_p = Path(f"{prefix}_objmodel.glb"), Path(f"{prefix}_objinfo.json")
    for p in (glb, info_p):  # the names Unity's ScanResponseParser.ModelSuffix / InfoSuffix expect
        if not p.exists() or p.stat().st_size == 0:
            problems.append(f"missing or empty output {p.name}")
    if problems:
        return None, lat, problems
    try:
        info = json.loads(info_p.read_text())
    except Exception as e:
        return None, lat, [f"{info_p.name} is not JSON: {e}"]
    problems += [f"objinfo: {x}" for x in G.objinfo_problems(info)]
    if not problems:
        w = G.unity_object_to_world(info, CAM_C2W)
        if not np.isfinite(w).all() or abs(np.linalg.det(w[:3, :3])) < 1e-9:
            problems.append("Unity ObjectToWorld composition is non-finite or singular")
    try:
        G.mesh_metrics(glb)
    except Exception as e:
        problems.append(f"glb unreadable: {e}")
    return {"mask": mask, "glb": glb, "objinfo": info}, lat, problems


def run_corpus(lib: Lib, addr: str, work: Path, args):
    seg = lib.session(TYPE_SEG, addr, args.connect_timeout)
    obj = lib.session(TYPE_OBJ, addr, args.connect_timeout)
    results = {}
    try:
        for case in G.CASES:
            results[case.name] = run_case(lib, seg, obj, case, work / case.name,
                                          args.segment_timeout, args.reconstruct_timeout)
    finally:
        lib.l.spatialblend_inference_session_destroy(seg)
        lib.l.spatialblend_inference_session_destroy(obj)
    return results


def load_golden(gdir: Path, name: str):
    d = gdir / name
    return {"mask": np.frombuffer((d / "mask.u8").read_bytes(), dtype=np.uint8).reshape(G.IMG_H, G.IMG_W),
            "glb": d / "mesh.glb", "objinfo": json.loads((d / "objinfo.json").read_text())}


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def cmd_record(args, lib, addr, kind):
    gdir = Path(args.golden_dir)
    thr = json.loads((HERE / "thresholds.json").read_text())
    runs = []
    for r in range(args.runs):
        keep = Path(tempfile.mkdtemp(prefix="golden_rec_"))
        runs.append((run_corpus(lib, addr, keep, args), keep))
    bad = [f"{c}: {p}" for c, (a, _, ps) in runs[0][0].items() for p in ps] or \
          [f"{c}: no artifacts" for c, (a, _, _) in runs[0][0].items() if a is None]
    if bad:
        print("RECORD FAILED:\n  " + "\n  ".join(bad), file=sys.stderr)
        return EXIT_SETUP
    for case in G.CASES:
        a = runs[0][0][case.name][0]
        d = gdir / case.name
        d.mkdir(parents=True, exist_ok=True)
        (d / "mask.u8").write_bytes(a["mask"].tobytes())
        shutil.copyfile(a["glb"], d / "mesh.glb")
        (d / "objinfo.json").write_text(json.dumps(a["objinfo"], indent=2))
        (d / "metrics.json").write_text(json.dumps(
            {"mesh": G.mesh_metrics(d / "mesh.glb"), "mask_fg_pixels": int(a["mask"].sum())}, indent=2))
    spread = {}
    for case in G.CASES:
        g = load_golden(gdir, case.name)
        for r in range(1, args.runs):
            a = runs[r][0][case.name][0]
            if a is None:
                continue
            for row in G.compare(case.name, a, g, thr):
                if isinstance(row["value"], (int, float)) and not isinstance(row["value"], bool):
                    spread.setdefault(row["metric"], []).append(row["value"])
    lats = {k: [runs[r][0][c.name][1][k] for r in range(args.runs) for c in G.CASES] for k in ("segment", "reconstruct")}
    (gdir / "calibration.json").write_text(json.dumps({
        "server_kind": kind, "addr": addr, "runs": args.runs,
        "run_to_run_vs_run0": {k: {"max": max(v), "n": len(v)} for k, v in spread.items()},
        "latency_s": {k: {"p50": pct(v, .5), "p95": pct(v, .95)} for k, v in lats.items()},
        "note": "Propose thresholds from this spread and review them before relying on them. Stub goldens prove the harness only.",
    }, indent=2))
    (gdir / "SERVER_KIND").write_text(kind + "\n")
    for _, keep in runs:
        shutil.rmtree(keep, ignore_errors=True)
    print(f"recorded {len(G.CASES)} goldens ({args.runs} runs, server_kind={kind}) in {gdir}")
    return EXIT_OK


def cmd_check(args, lib, addr, kind):
    gdir = Path(args.golden_dir)
    if not (gdir / "SERVER_KIND").exists():
        print(f"no golden in {gdir}; run `record` against a trusted server first", file=sys.stderr)
        return EXIT_SETUP
    gkind = (gdir / "SERVER_KIND").read_text().strip()
    if gkind != kind:
        print(f"golden was recorded against server_kind={gkind}, this run is {kind}", file=sys.stderr)
        return EXIT_SETUP
    thr = json.loads((HERE / "thresholds.json").read_text())
    rows, fails = [], []
    with tempfile.TemporaryDirectory(prefix="golden_chk_") as w:
        res = run_corpus(lib, addr, Path(w), args)
        for case in G.CASES:
            art, lat, problems = res[case.name]
            fails += [f"{case.name}: {p}" for p in problems]
            for k, lim in thr["latency_p95_s"].items():
                ok = lat.get(k, 1e9) <= lim
                rows.append({"case": case.name, "metric": f"{k}_latency_s", "value": round(lat.get(k, -1), 2),
                             "limit": f"<= {lim}", "ok": ok})
            if art is not None:
                rows += G.compare(case.name, art, load_golden(gdir, case.name), thr)
    fails += [f"{r['case']}: {r['metric']}={r['value']} (limit {r['limit']})" for r in rows if not r["ok"]]
    report = {"server_kind": kind, "addr": addr, "ok": not fails, "rows": rows, "failures": fails}
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))
    for r in rows:
        print(f"{'ok  ' if r['ok'] else 'FAIL'} {r['case']:8s} {r['metric']:28s} {r['value']!s:>10s}  {r['limit']}")
    if fails:
        print("\nDRIFT:\n  " + "\n  ".join(fails), file=sys.stderr)
        return EXIT_DRIFT
    print("\ngolden pipeline: no drift")
    return EXIT_OK


def free_port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def cmd_selftest(args, lib):
    port = free_port()
    p = subprocess.Popen([sys.executable, str(HERE / "stub_server.py"), "--port", str(port)],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        line = p.stdout.readline()
        if "listening" not in line:
            print("stub server failed to start: " + line + (p.stdout.read() or ""), file=sys.stderr)
            return EXIT_SETUP
        addr = f"127.0.0.1:{port}"
        with tempfile.TemporaryDirectory(prefix="golden_self_") as g:
            args.golden_dir = g
            args.runs = max(args.runs, 2)
            rc = cmd_record(args, lib, addr, "stub")
            if rc:
                return rc
            rc = cmd_check(args, lib, addr, "stub")
            if rc:
                return rc
            # The harness must also FAIL when the output drifts: corrupt one golden and expect exit 1.
            m = Path(g) / "disk" / "mask.u8"
            raw = bytearray(m.read_bytes())
            for i in range(0, len(raw), 2):
                raw[i] ^= 1
            m.write_bytes(bytes(raw))
            rc = cmd_check(args, lib, addr, "stub")
            if rc != EXIT_DRIFT:
                print(f"selftest: a corrupted golden gave exit {rc}, expected {EXIT_DRIFT}", file=sys.stderr)
                return EXIT_SETUP
            print("selftest: drift detection works (corrupted golden -> exit 1)")
            return EXIT_OK
    finally:
        p.terminate()
        try:
            p.wait(5)
        except subprocess.TimeoutExpired:
            p.kill()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["check", "record", "selftest"])
    ap.add_argument("--lib", default=os.environ.get("SPATIALBLEND_SESSION_LIB"),
                    help="path to libspatialblend_session.{so,dylib,dll} (or $SPATIALBLEND_SESSION_LIB)")
    ap.add_argument("--addr", default=os.environ.get("SPATIALBLEND_SAM3D_ADDR"), help="host:port of the scan server")
    ap.add_argument("--golden-dir", default=os.environ.get("SPATIALBLEND_GOLDEN_DIR"))
    ap.add_argument("--server-kind", default="real", choices=["real", "stub"])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--report")
    ap.add_argument("--connect-timeout", type=float, default=20)
    ap.add_argument("--segment-timeout", type=float, default=120)
    ap.add_argument("--reconstruct-timeout", type=float, default=600)
    args = ap.parse_args()
    try:
        if not args.lib:
            raise Setup("--lib or $SPATIALBLEND_SESSION_LIB is required")
        lib = Lib(args.lib)
        if args.mode == "selftest":
            return cmd_selftest(args, lib)
        if not args.addr or not args.golden_dir:
            raise Setup("--addr and --golden-dir are required")
        return (cmd_record if args.mode == "record" else cmd_check)(args, lib, args.addr, args.server_kind)
    except Setup as e:
        print(f"setup error: {e}", file=sys.stderr)
        return EXIT_SETUP


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)  # the library's worker threads outlive main; do not wait on them
