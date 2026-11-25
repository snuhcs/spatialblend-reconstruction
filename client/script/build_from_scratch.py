#!/usr/bin/env python3
# Copyright (c) 2026 Seoul National University
# SPDX-License-Identifier: MIT

"""Build libspatialblend_session from a fresh clone, the way the release workflow does (.github/workflows/client-release.yml).

    python3 script/build_from_scratch.py                       # this host's library (macOS, Linux or Windows)
    python3 script/build_from_scratch.py --platform android    # Android arm64 (needs ANDROID_NDK_HOME, NDK r27d)
    python3 script/build_from_scratch.py --platform host --platform android --test

Run it from anywhere; it works in client/. Prerequisites (checked first):
  * Bazel through Bazelisk (`bazel` on PATH); Bazelisk downloads the version in .bazelversion (8.4.2).
  * A C++17 toolchain: Xcode command line tools (macOS), gcc or clang (Linux), Visual Studio 2022 with C++ (Windows).
  * Python 3 (Bazel's rules_python brings its own interpreter for the build).
  * Android only: the NDK, r27d as in CI, in ANDROID_NDK_HOME.
No secret is needed: the library holds no credentials; the server address and TLS choice are set by the app at run time.

Each platform's library goes to output/<platform>/ (git-ignored) with the headers, after the export audit
(script/audit_session_plugin_exports.py) passes and writes SHA256SUMS. --output-base gives Bazel its own output base
(delete it afterwards to reclaim the disk); --disk-cache shares a cache between clones.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

CLIENT = Path(__file__).resolve().parents[1]
TARGET = "//session/c:libspatialblend_session_c"
TAR = CLIENT / "bazel-bin/session/c/libspatialblend_session_c.tar"
NDK_MAJOR = 27


def host() -> str:
    system = platform.system()
    return {"Darwin": "macos", "Linux": "linux", "Windows": "windows"}.get(system, system.lower())


def fail(message: str) -> None:
    print(f"build_from_scratch: {message}", file=sys.stderr)
    raise SystemExit(2)


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    result = subprocess.run(command, cwd=CLIENT)
    if result.returncode:
        fail(f"command failed ({result.returncode}): {' '.join(command)}")


def check_prerequisites(platforms: list[str]) -> str:
    bazel = shutil.which("bazelisk") or shutil.which("bazel")
    if not bazel:
        fail("Bazel not found: install Bazelisk (https://github.com/bazelbuild/bazelisk) as `bazel`")
    wanted = (CLIENT / ".bazelversion").read_text().strip()
    version = subprocess.run([bazel, "--version"], cwd=CLIENT, capture_output=True, text=True).stdout.strip()
    if wanted not in version:
        fail(f"Bazel in client/ is {version or 'unknown'}, .bazelversion wants {wanted}: use Bazelisk, which honours it")
    print(f"bazel: {version}")
    if host() == "macos" and subprocess.run(["xcrun", "--find", "clang"], capture_output=True).returncode:
        fail("no Xcode command line tools: run xcode-select --install")
    if "android" in platforms:
        ndk = os.environ.get("ANDROID_NDK_HOME")
        if not ndk or not Path(ndk).is_dir():
            fail(f"--platform android needs ANDROID_NDK_HOME pointing at an Android NDK (r{NDK_MAJOR}d as in CI)")
        props = Path(ndk) / "source.properties"
        match = re.search(r"Pkg\.Revision\s*=\s*(\d+)", props.read_text()) if props.is_file() else None
        if not match:
            fail(f"{ndk} has no source.properties: not an NDK")
        if int(match.group(1)) != NDK_MAJOR:
            print(f"warning: NDK {match.group(1)} found; CI and the releases use r{NDK_MAJOR}d", file=sys.stderr)
    return bazel


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dumpbin() -> str:
    """dumpbin.exe of the newest Visual Studio with C++ (as client-release.yml finds it), for the Windows export audit."""
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
    vs = subprocess.run([str(vswhere), "-latest", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                         "-property", "installationPath"], capture_output=True,
                        text=True).stdout.strip() if vswhere.is_file() else ""
    found = sorted(Path(vs).glob("VC/Tools/MSVC/*/bin/Hostx64/x64/dumpbin.exe")) if vs else []
    if not found:
        fail("dumpbin.exe not found: install Visual Studio 2022 with 'Desktop development with C++' (the export audit needs it)")
    return str(found[-1])


def nm_for(target: str) -> str | None:
    if target != "android":
        return None
    found = glob.glob(os.path.join(os.environ["ANDROID_NDK_HOME"], "toolchains/llvm/prebuilt/*/bin/llvm-nm*"))
    return found[0] if found else None


def build(bazel: str, startup: list[str], common: list[str], target: str, test: bool, out_root: Path) -> Path:
    out = out_root / target
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    if target == "windows":
        # As CI: a short output root keeps protobuf/gRPC paths under MAX_PATH; SPATIALBLEND_BUILD_DLL selects dllexport.
        startup = startup or ["--output_user_root=C:/b"]
        if test:
            run([bazel, *startup, "test", *common, "--test_output=errors", "//session/..."])
        run([bazel, *startup, "build", *common, "--copt=-DSPATIALBLEND_BUILD_DLL", "--strip=always", "//session/c:c_shared"])
        library = out / "libspatialblend_session.dll"
        shutil.copy2(CLIENT / "bazel-bin/session/c/libspatialblend_session.dll", library)
        library.chmod(0o644)   # Bazel outputs are read-only
        include = out / "include/spatialblend_session"   # the layout of the tar the other platforms extract
        include.mkdir(parents=True)
        for header in sorted((CLIENT / "session/c").glob("*.h")):
            shutil.copy2(header, include / header.name)
        tool = ["--dumpbin", dumpbin()]
    else:
        config = ["--config=android"] if target == "android" else []
        if test and target != "android":
            run([bazel, *startup, "test", *common, "--test_output=errors", "//session/..."])
        run([bazel, *startup, "build", *common, *config, TARGET])
        with tarfile.open(TAR) as archive:
            archive.extractall(out, filter="data") if sys.version_info >= (3, 12) else archive.extractall(out)
        name = "libspatialblend_session.dylib" if target == "macos" else "libspatialblend_session.so"
        library = out / name
        shutil.copy2(out / "lib" / name, library)
        nm = nm_for(target)
        tool = ["--nm", nm] if nm else []
    run([sys.executable, "script/audit_session_plugin_exports.py", "--library", str(library), "--headers", "session/c",
         "--summary", str(out / "summary.txt"), "--checksum", str(out / "SHA256SUMS"), *tool])
    return library


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--platform", action="append", choices=["host", "android"],
                        help="what to build (repeatable; default host)")
    parser.add_argument("--test", action="store_true", help="also run bazel test //session/... (host only)")
    parser.add_argument("--output-base", help="Bazel --output_base for this clone (delete it afterwards)")
    parser.add_argument("--disk-cache", help="Bazel --disk_cache directory shared between builds")
    parser.add_argument("--out", type=Path, default=CLIENT / "output", help="where the libraries go (default client/output)")
    args = parser.parse_args(argv)
    platforms = args.platform or ["host"]
    bazel = check_prerequisites(platforms)
    startup = [f"--output_base={Path(args.output_base).resolve()}"] if args.output_base else []
    common = [f"--disk_cache={Path(args.disk_cache).resolve()}"] if args.disk_cache else []
    results = []
    for choice in platforms:
        target = host() if choice == "host" else "android"
        results.append((target, build(bazel, startup, common, target, args.test, args.out.resolve())))
    print("\nbuilt:")
    for target, library in results:
        print(f"  {target:8} {library}  sha256 {sha256(library)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
