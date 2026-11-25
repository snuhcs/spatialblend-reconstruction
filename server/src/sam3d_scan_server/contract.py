# Copyright (c) 2026 Seoul National University
# SPDX-License-Identifier: MIT

"""The wire contract of a scan proto: package, services, methods and every message field (name, number, type, label).

Comments and formatting are not part of it. A client that keeps its own copy of distributed.proto checks it with

    python -m sam3d_scan_server.contract path/to/client/distributed.proto

which exits 0 when the copy describes the same wire format as this server's proto, and prints the differences
otherwise.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import List, Optional


def describe(proto: Path) -> dict:
    """Compiles one .proto (grpc_tools' protoc, no network) and returns its contract as plain data."""
    from google.protobuf import descriptor_pb2
    from grpc_tools import protoc

    proto = Path(proto).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "set.pb"
        rc = protoc.main(["protoc", f"-I{proto.parent}", f"--descriptor_set_out={out}", str(proto)])
        if rc != 0:
            raise RuntimeError(f"protoc failed for {proto} (exit {rc})")
        fds = descriptor_pb2.FileDescriptorSet.FromString(out.read_bytes())
    (f,) = [x for x in fds.file if x.name == proto.name]
    ftype = descriptor_pb2.FieldDescriptorProto
    messages, enums = {}, {}

    def walk(prefix, message_types, enum_types):
        for e in enum_types:
            enums[prefix + e.name] = {v.name: v.number for v in e.value}
        for m in message_types:
            fields = {}
            for fd in m.field:
                t = (fd.number, ftype.Type.Name(fd.type), ftype.Label.Name(fd.label), fd.type_name)
                if fd.HasField("oneof_index") and not fd.proto3_optional:
                    t += ("oneof:" + m.oneof_decl[fd.oneof_index].name,)
                if fd.proto3_optional:
                    t += ("optional",)
                fields[fd.name] = t
            messages[prefix + m.name] = fields
            walk(prefix + m.name + ".", m.nested_type, m.enum_type)

    walk("", f.message_type, f.enum_type)
    out = {
        "package": f.package,
        "syntax": f.syntax or "proto2",
        "services": {s.name: {m.name: (m.input_type, m.output_type, m.client_streaming, m.server_streaming)
                              for m in s.method} for s in f.service},
        "messages": messages,
    }
    if enums:
        out["enums"] = enums
    return out


def differences(a: dict, b: dict, path: str = "") -> List[str]:
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            where = f"{path}.{k}" if path else k
            if k not in a:
                out.append(f"only in the other proto: {where} = {b[k]}")
            elif k not in b:
                out.append(f"only in this server's proto: {where} = {a[k]}")
            else:
                out += differences(a[k], b[k], where)
    elif a != b:
        out.append(f"{path}: {a} != {b}")
    return out


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    from .protocol import proto_path
    diff = differences(describe(proto_path()), describe(Path(argv[0])))
    for line in diff:
        print(line)
    print("same wire contract" if not diff else f"{len(diff)} difference(s)")
    return 0 if not diff else 1


if __name__ == "__main__":
    sys.exit(main())
