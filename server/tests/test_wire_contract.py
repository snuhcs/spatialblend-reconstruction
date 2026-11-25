# Copyright (c) 2026 Seoul National University
# SPDX-License-Identifier: MIT

"""The wire contract the deployed clients depend on. A change here is a breaking API change: it needs a new package
version (spatialblend.scan.v2) rather than an edit."""
from pathlib import Path

from sam3d_scan_server.contract import describe, differences, main
from sam3d_scan_server.protocol import proto_path

P = ".spatialblend.scan.v1."

EXPECTED = {
    "package": "spatialblend.scan.v1",
    "syntax": "proto3",
    "services": {"ReconstructionService": {
        "Segment": (P + "SegmentRequest", P + "SegmentResponse", False, False),
        "Reconstruct": (P + "ReconstructRequest", P + "ReconstructResponse", False, False),
    }},
    "messages": {
        "SegPoint": {"x": (1, "TYPE_FLOAT", "LABEL_OPTIONAL", ""), "y": (2, "TYPE_FLOAT", "LABEL_OPTIONAL", ""),
                     "z": (3, "TYPE_FLOAT", "LABEL_OPTIONAL", "")},
        "SegmentRequest": {"client_id": (1, "TYPE_STRING", "LABEL_OPTIONAL", ""),
                           "image_data": (2, "TYPE_BYTES", "LABEL_OPTIONAL", ""),
                           "points": (3, "TYPE_MESSAGE", "LABEL_REPEATED", P + "SegPoint")},
        "SegmentResponse": {"success": (1, "TYPE_BOOL", "LABEL_OPTIONAL", ""),
                            "message": (2, "TYPE_STRING", "LABEL_OPTIONAL", ""),
                            "mask_data": (3, "TYPE_BYTES", "LABEL_OPTIONAL", ""),
                            "mask_width": (4, "TYPE_INT32", "LABEL_OPTIONAL", ""),
                            "mask_height": (5, "TYPE_INT32", "LABEL_OPTIONAL", "")},
        "ReconstructRequest": {"client_id": (1, "TYPE_STRING", "LABEL_OPTIONAL", ""),
                               "image_data": (2, "TYPE_BYTES", "LABEL_OPTIONAL", ""),
                               "mask_data": (3, "TYPE_BYTES", "LABEL_OPTIONAL", ""),
                               "mask_width": (4, "TYPE_INT32", "LABEL_OPTIONAL", ""),
                               "mask_height": (5, "TYPE_INT32", "LABEL_OPTIONAL", "")},
        "ReconstructResponse": {"success": (1, "TYPE_BOOL", "LABEL_OPTIONAL", ""),
                                "message": (2, "TYPE_STRING", "LABEL_OPTIONAL", ""),
                                "glb_file": (3, "TYPE_BYTES", "LABEL_OPTIONAL", ""),
                                "json_file": (4, "TYPE_BYTES", "LABEL_OPTIONAL", "")},
    },
}


def test_the_proto_is_the_pinned_wire_contract():
    assert differences(describe(proto_path()), EXPECTED) == []


def test_the_generated_stub_uses_the_published_method_paths():
    from sam3d_scan_server import protocol
    pb, pb_grpc = protocol.load()
    import inspect
    src = inspect.getsource(pb_grpc)
    assert "/spatialblend.scan.v1.ReconstructionService/Segment" in src
    assert "/spatialblend.scan.v1.ReconstructionService/Reconstruct" in src


def test_contract_check_ignores_comments_and_reports_a_changed_field(tmp_path: Path, capsys):
    text = proto_path().read_text()
    same = tmp_path / "same" / "distributed.proto"
    same.parent.mkdir()
    same.write_text("// a client's copy with other comments\n" + text.replace("// Phase 1", "// first phase"))
    assert main([str(same)]) == 0

    changed = tmp_path / "changed" / "distributed.proto"
    changed.parent.mkdir()
    changed.write_text(text.replace("int32 mask_width = 4;", "int64 mask_width = 4;", 1))
    assert main([str(changed)]) == 1
    assert "SegmentResponse.mask_width" in capsys.readouterr().out


def test_one_proto_for_client_and_server():
    """In a repository checkout, proto/distributed.proto is the only copy: the client (client/, Bazel) compiles it
    through @scan_proto and this package loads it, so the two sides cannot drift."""
    import subprocess
    import pytest
    root = Path(__file__).resolve().parents[2]
    if not (root / ".git").exists() or not (root / "client" / "MODULE.bazel").is_file():
        pytest.skip("not inside the repository checkout")
    tracked = subprocess.run(["git", "-C", str(root), "ls-files", "*distributed.proto"], capture_output=True,
                             text=True, check=True).stdout.split()
    assert tracked == ["proto/distributed.proto"]
    assert proto_path() == root / "proto" / "distributed.proto"
    assert 'path = "../proto"' in (root / "client" / "MODULE.bazel").read_text()
    assert "@scan_proto//:distributed.proto" in (
        root / "client" / "session" / "inference" / "distributed" / "proto" / "BUILD").read_text()
