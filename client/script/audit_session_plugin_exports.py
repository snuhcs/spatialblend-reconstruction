#!/usr/bin/env python3
# Copyright (c) 2026 Seoul National University
# SPDX-License-Identifier: MIT

"""Audit a libspatialblend_session shared library for the public spatialblend_* exports.

The expected names are the functions declared with SPATIALBLEND_EXPORT in
session/c. Platform tools:

  ELF (.so)       nm -D --defined-only
  Mach-O (.dylib) nm -gU
  COFF (.dll)     dumpbin /exports

Linker section markers (__start_* / __stop_*) are ignored. Any other exported
symbol outside spatialblend_* fails the audit, which is what the Linux version
script and the macOS exported-symbols list are there to prevent.

SHA256SUMS is written only when the audit passes, so a library that failed the
audit is never published with a checksum.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

EXPORT_MACROS = ("SPATIALBLEND_EXPORT",)
PREFIX = "spatialblend_"
# The node itself: "VERS_0.0.1" (GNU nm) or "VERS_0.0.1@@VERS_0.0.1" (llvm-nm appends its own version).
VERSION_NODE = re.compile(r"^(VERS_[0-9A-Za-z_.]*)(?:@@?\1)?$")
LINKER_BOUNDARY = re.compile(r"^__(?:start|stop)_")
DUMPBIN_EXPORT = re.compile(r"^\s+\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]{8,}\s+(\S+)")


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//.*?$", "", text, flags=re.M)


def expected_symbols(headers_dir: Path) -> set[str]:
    """Return spatialblend_* function names declared for export."""
    names: set[str] = set()
    headers = sorted(headers_dir.glob("*.h"))
    if not headers:
        raise SystemExit(f"no headers in {headers_dir}")
    for path in headers:
        text = _strip_comments(path.read_text(encoding="utf-8"))
        text = "\n".join(
            line for line in text.splitlines() if not line.lstrip().startswith("#")
        )
        for macro in EXPORT_MACROS:
            for match in re.finditer(rf"\b{macro}\b", text):
                decl = text[match.end():]
                semi = decl.find(";")
                if semi < 0:
                    continue
                names.update(
                    re.findall(rf"\b({PREFIX}[A-Za-z0-9_]+)\s*\(", decl[:semi])
                )
    if not names:
        raise SystemExit(f"no exported {PREFIX}* declarations in {headers_dir}")
    return names


def normalize_export(name: str) -> str:
    """Drop a GNU symbol version and the Mach-O underscore prefix."""
    name = name.split("@@", 1)[0].split("@", 1)[0]
    if name.startswith("_" + PREFIX):
        return name[1:]
    return name


def parse_nm(text: str) -> set[str]:
    """Return the names of defined, externally visible symbols from nm output.

    Not exports, so skipped:
      U            undefined (imports)
      b d t r ...  lowercase types are STB_LOCAL. Newer GNU ld (2.42) copies
                   thread-local statics of the static deps (grpc, absl,
                   protobuf, re2) into .dynsym as LOCAL entries to carry TLS
                   relocations; the dynamic linker cannot bind or dlsym them.
                   w, v and u are lowercase but weak/unique-global, so kept.
      A VERS_*     the version-definition node of the version script (llvm-nm prints it as
                   VERS_x@@VERS_x).
    """
    names: set[str] = set()
    for line in text.splitlines():
        parts = line.split()
        # "address TYPE name" or "TYPE name". Undefined symbols are type U.
        if len(parts) >= 2 and len(parts[-2]) == 1 and parts[-2].isalpha():
            kind, name = parts[-2], parts[-1]
            if kind.upper() == "U":
                continue
            if kind.islower() and kind not in "wvu":
                continue
            if kind == "A" and VERSION_NODE.match(name):
                continue
            names.add(name)
    return names


def parse_dumpbin(text: str) -> set[str]:
    names: set[str] = set()
    for line in text.splitlines():
        match = DUMPBIN_EXPORT.match(line)
        if match:
            names.add(match.group(1))
    return names


def audit_symbols(expected: set[str], exported: set[str]) -> tuple[list[str], list[str]]:
    """Return (missing expected names, unexpected non-spatialblend_* names)."""
    normalized = {normalize_export(name) for name in exported}
    missing = sorted(expected - normalized)
    foreign = sorted(
        name
        for name in normalized
        if not name.startswith(PREFIX) and not LINKER_BOUNDARY.match(name)
    )
    return missing, foreign


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_sha256sums(library: Path, checksum_path: Path) -> str:
    digest = sha256_file(library)
    checksum_path.write_text(f"{digest}  {library.name}\n", encoding="utf-8")
    return digest


def _run(command: list[str]) -> str:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise SystemExit(f"missing tool: {command[0]}") from exc
    if completed.returncode != 0:
        raise SystemExit(
            f"{' '.join(command)} failed ({completed.returncode}): {completed.stderr}"
        )
    return completed.stdout


def list_exports(library: Path, dumpbin: str | None, nm: str | None) -> set[str]:
    suffix = library.suffix.lower()
    if suffix == ".dll":
        return parse_dumpbin(_run([dumpbin or "dumpbin", "/exports", str(library)]))
    if suffix == ".dylib":
        return parse_nm(_run([nm or "nm", "-gU", str(library)]))
    if suffix == ".so":
        # Android arm64 is audited with the NDK llvm-nm, not host binutils nm.
        return parse_nm(_run([nm or "nm", "-D", "--defined-only", str(library)]))
    raise SystemExit(f"unsupported library type: {library}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--headers", type=Path, default=Path("session/c"))
    parser.add_argument("--dumpbin", default=None, help="Path to dumpbin.exe")
    parser.add_argument("--nm", default=None, help="nm or llvm-nm for .so and .dylib")
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument(
        "--checksum",
        type=Path,
        default=None,
        help="Write a sha256sum sidecar for this library, only if the audit passes",
    )
    args = parser.parse_args(argv)

    library = args.library
    if not library.is_file():
        print(f"library not found: {library}", file=sys.stderr)
        return 1

    expected = expected_symbols(args.headers)
    exported = list_exports(library, args.dumpbin, args.nm)
    missing, foreign = audit_symbols(expected, exported)
    normalized = {normalize_export(name) for name in exported}
    own = sum(1 for name in normalized if name.startswith(PREFIX))
    digest = sha256_file(library)
    summary = (
        f"sha256={digest} exports={len(normalized)} spatialblend={own} "
        f"expected={len(expected)} missing={len(missing)} other={len(foreign)}"
    )
    print(summary)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        lines = [summary]
        if missing:
            lines += ["missing:"] + [f"  {name}" for name in missing]
        if foreign:
            lines += ["foreign:"] + [f"  {name}" for name in foreign]
        args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if missing or foreign:
        for name in missing[:40]:
            print(f"missing {name}", file=sys.stderr)
        for name in foreign[:40]:
            print(f"foreign {name}", file=sys.stderr)
        return 1
    if args.checksum:
        write_sha256sums(library, args.checksum)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
