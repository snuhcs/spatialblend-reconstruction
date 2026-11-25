#!/usr/bin/env python3
# Copyright (c) 2026 Seoul National University
# SPDX-License-Identifier: MIT

"""Tests for the export audit. No bazel, no native tools, no network."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit_session_plugin_exports as audit  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HEADERS = ROOT / "session" / "c"


class AuditTest(unittest.TestCase):
    def test_expected_symbols_come_from_the_header(self) -> None:
        names = audit.expected_symbols(HEADERS)
        for name in (
            "spatialblend_inference_session_create",
            "spatialblend_inference_sam3d",
            "spatialblend_inference_segment",
            "spatialblend_inference_reconstruct",
            "spatialblend_cleanup_all",
        ):
            self.assertIn(name, names)
        self.assertTrue(all(n.startswith("spatialblend_") for n in names))

    def test_parse_nm_elf_and_macho(self) -> None:
        elf = "0000000000323e04 T spatialblend_cleanup_all@@VERS_0.0.1\n                 U memcpy\n"
        macho = "0000000000001000 T _spatialblend_cleanup_all\n"
        self.assertEqual(
            {audit.normalize_export(n) for n in audit.parse_nm(elf)}, {"spatialblend_cleanup_all"}
        )
        self.assertEqual(
            {audit.normalize_export(n) for n in audit.parse_nm(macho)}, {"spatialblend_cleanup_all"}
        )

    def test_parse_nm_skips_local_symbols_and_version_node(self) -> None:
        text = (
            "0000000000000000 A VERS_0.0.1\n"
            "0000000000000110 b _ZN9grpc_core7ExecCtx9exec_ctx_E\n"
            "0000000000000020 d _ZN6google8protobuf8internal15ThreadSafeArena13thread_cache_E\n"
            "0000000000323e04 T spatialblend_cleanup_all@@VERS_0.0.1\n"
            "0000000000001000 W weak_global\n"
            "0000000000001000 w weak_lower\n"
            "0000000000002000 B real_global_bss\n"
            "0000000000002000 A some_abs\n"
        )
        self.assertEqual(
            audit.parse_nm(text),
            {
                "spatialblend_cleanup_all@@VERS_0.0.1",
                "weak_global",
                "weak_lower",
                "real_global_bss",
                "some_abs",
            },
        )

    def test_parse_nm_skips_llvm_nm_version_node(self) -> None:
        text = (
            "0000000000000000 A VERS_0.0.1@@VERS_0.0.1\n"
            "0000000000323e04 T spatialblend_cleanup_all@@VERS_0.0.1\n"
            "0000000000000000 A VERS_0.0.1@@VERS_9.9.9\n"
        )
        self.assertEqual(
            audit.parse_nm(text),
            {"spatialblend_cleanup_all@@VERS_0.0.1", "VERS_0.0.1@@VERS_9.9.9"},
        )

    def test_parse_dumpbin(self) -> None:
        text = "    ordinal hint RVA      name\n\n          1    0 0000A4A0 spatialblend_cleanup_all\n"
        self.assertEqual(audit.parse_dumpbin(text), {"spatialblend_cleanup_all"})

    def test_missing_and_foreign_symbols_fail(self) -> None:
        expected = {"spatialblend_a", "spatialblend_b"}
        missing, foreign = audit.audit_symbols(expected, {"spatialblend_a", "AES_encrypt"})
        self.assertEqual(missing, ["spatialblend_b"])
        self.assertEqual(foreign, ["AES_encrypt"])
        missing, foreign = audit.audit_symbols(
            expected, {"spatialblend_a", "spatialblend_b", "__start_pb_defaults"}
        )
        self.assertEqual((missing, foreign), ([], []))

    def test_checksum_only_written_when_the_audit_passes(self) -> None:
        names = sorted(audit.expected_symbols(HEADERS))
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            lib = tmp_path / "libspatialblend_session.so"
            lib.write_bytes(b"fake")
            sums = tmp_path / "SHA256SUMS"
            original = audit.list_exports
            try:
                audit.list_exports = lambda *a, **k: set(names) | {"AES_encrypt"}
                code = audit.main(["--library", str(lib), "--headers", str(HEADERS), "--checksum", str(sums)])
                self.assertEqual(code, 1)
                self.assertFalse(sums.exists())
                audit.list_exports = lambda *a, **k: set(names[:-1])
                code = audit.main(["--library", str(lib), "--headers", str(HEADERS), "--checksum", str(sums)])
                self.assertEqual(code, 1)
                self.assertFalse(sums.exists())
                audit.list_exports = lambda *a, **k: set(names)
                code = audit.main(["--library", str(lib), "--headers", str(HEADERS), "--checksum", str(sums)])
                self.assertEqual(code, 0)
                digest, name = sums.read_text().split()
                self.assertEqual(name, lib.name)
                self.assertEqual(digest, audit.sha256_file(lib))
            finally:
                audit.list_exports = original


if __name__ == "__main__":
    unittest.main()
