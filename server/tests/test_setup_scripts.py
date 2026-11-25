# Copyright (c) 2026 Seoul National University
# SPDX-License-Identifier: MIT

"""scripts/setup_server.sh and scripts/configure_secrets.py: where the Hugging Face token goes, and that no value is printed.

setup_server.sh runs against shim programs (python3, python, uv, pip, nvidia-smi, uname, sam3d-scan-models,
sam3d-scan-server, install_sam3d_objects.sh) that only log, per call, whether HF_TOKEN / HUGGING_FACE_HUB_TOKEN were in
their environment (and whether HF_TOKEN equals the expected dummy), never the value. No GPU, no network.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

import pytest

SERVER = Path(__file__).resolve().parents[1]
DUMMY = "hf_" + "Dummy0Token1For2Tests3Only4"   # shape of a token, not a token

SHIM = r"""#!/usr/bin/env bash
name=$(basename "$0")
hf=unset; [[ -n ${HF_TOKEN:-} ]] && hf=set
[[ -n ${HF_TOKEN:-} && ${HF_TOKEN} == "$SHIM_EXPECT" ]] && hf=match
hub=unset; [[ -n ${HUGGING_FACE_HUB_TOKEN:-} ]] && hub=set
printf '%s|%s|hf=%s|hub=%s\n' "$name" "${1:-}" "$hf" "$hub" >> "$SHIM_LOG"
case "$name" in
  uname) echo Linux ;;
  uv) dir=${@: -1}; mkdir -p "$dir/bin"; cp "$0" "$dir/bin/python"; : > "$dir/bin/activate" ;;
esac
exit 0
"""

pytestmark = pytest.mark.skipif(os.name == "nt" or shutil.which("bash") is None, reason="needs bash (POSIX shims)")


def make_tree(tmp: Path) -> tuple[Path, Path, Path]:
    server = tmp / "server"
    (server / "scripts").mkdir(parents=True)
    shutil.copy(SERVER / "scripts/setup_server.sh", server / "scripts/setup_server.sh")
    shims = tmp / "shims"
    shims.mkdir()
    for name in ("python3", "python", "uv", "pip", "nvidia-smi", "uname", "sam3d-scan-models", "sam3d-scan-server"):
        (shims / name).write_text(SHIM)
    (server / "scripts/install_sam3d_objects.sh").write_text(SHIM)
    for path in [*shims.iterdir(), server / "scripts/install_sam3d_objects.sh", server / "scripts/setup_server.sh"]:
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return server, shims, tmp / "shim.log"


def run_setup(tmp: Path, args: list[str], env_extra: dict[str, str], dotenv: str | None = None):
    server, shims, log = make_tree(tmp)
    if dotenv is not None:
        (server / ".env").write_bytes(dotenv.encode())
    env = {"PATH": f"{shims}:/usr/bin:/bin", "HOME": str(tmp), "SHIM_LOG": str(log), "SHIM_EXPECT": DUMMY,
           "SAM3D_OBJECTS_ROOT": str(tmp / "sam3d"), **env_extra}
    result = subprocess.run(["bash", str(server / "scripts/setup_server.sh"), *args], env=env, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=60)
    calls = [line.split("|") for line in log.read_text().splitlines()] if log.is_file() else []
    return result, calls


def token_seen(calls) -> dict[str, set[str]]:
    seen: dict[str, set[str]] = {}
    for name, arg, hf, hub in calls:
        key = f"{name} {arg}".strip()
        seen.setdefault(key, set()).add(f"{hf},{hub}")
    return seen


@pytest.mark.parametrize("source", ["env", "dotenv-crlf"])
def test_the_token_reaches_only_the_check_and_the_download(tmp_path, source):
    env_extra = {"HF_TOKEN": DUMMY} if source == "env" else {}
    dotenv = f"HF_TOKEN={DUMMY}   \r\nSAM3D_SCAN_PORT=9811\r\n" if source != "env" else None
    result, calls = run_setup(tmp_path, ["--run"], env_extra, dotenv)
    assert result.returncode == 0, result.stderr
    assert DUMMY not in result.stdout + result.stderr
    downloads = [c for c in calls if c[0] == "sam3d-scan-models"]
    assert downloads and all(c[2] == "hf=match" for c in downloads), downloads
    checks = [c for c in calls if c[0] == "python3" and c[1].endswith("configure_secrets.py")]
    if source == "env":
        assert checks and all(c[2] == "hf=match" for c in checks), checks
    others = [c for c in calls if c not in downloads and c not in checks]
    assert others, "expected pip, install, server and probe calls"
    leaked = [c for c in others if c[2] != "hf=unset" or c[3] != "hub=unset"]
    assert not leaked, leaked
    assert any(c[0] == "sam3d-scan-server" for c in others)
    assert any(c[0] == "install_sam3d_objects.sh" for c in others)


def test_the_hub_token_name_is_isolated_too(tmp_path):
    result, calls = run_setup(tmp_path, ["--run"], {"HUGGING_FACE_HUB_TOKEN": DUMMY})
    assert result.returncode == 0, result.stderr
    for name, arg, hf, hub in calls:
        expected = "hub=set" if name == "sam3d-scan-models" or arg.endswith("configure_secrets.py") else "hub=unset"
        assert hub == expected, (name, arg, hub)


def test_cpu_mode_never_hands_out_the_token(tmp_path):
    result, calls = run_setup(tmp_path, ["--cpu"], {"HF_TOKEN": DUMMY})
    assert result.returncode == 0, result.stderr
    leaked = [c for c in calls if not c[1].endswith("configure_secrets.py") and (c[2] != "hf=unset" or c[3] != "hub=unset")]
    assert not leaked, leaked
    assert not any(c[0] == "sam3d-scan-models" for c in calls)


# --------------------------------------------------------------------------------------------------------------------
# configure_secrets.py
# --------------------------------------------------------------------------------------------------------------------

def run_secrets(tmp: Path, args: list[str], env_extra: dict[str, str] | None = None):
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp), "HF_HOME": str(tmp / "hf"), **(env_extra or {})}
    return subprocess.run([sys.executable, str(SERVER / "scripts/configure_secrets.py"), *args], env=env,
                          stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60)


def test_check_with_nothing_set_fails_and_names_the_token(tmp_path):
    result = run_secrets(tmp_path, ["--check", "--env-file", str(tmp_path / "none.env")])
    assert result.returncode == 2
    assert "HF_TOKEN" in result.stdout and "MISSING" in result.stdout


def test_an_environment_token_counts_but_is_written_only_with_save(tmp_path):
    env_file = tmp_path / "server.env"
    result = run_secrets(tmp_path, ["--non-interactive", "--env-file", str(env_file)], {"HF_TOKEN": DUMMY})
    assert result.returncode == 0, result.stderr
    assert DUMMY not in result.stdout + result.stderr
    assert not env_file.exists()
    result = run_secrets(tmp_path, ["--non-interactive", "--save", "--env-file", str(env_file)], {"HF_TOKEN": DUMMY})
    assert result.returncode == 0, result.stderr
    assert DUMMY not in result.stdout + result.stderr
    assert f"HF_TOKEN={DUMMY}" in env_file.read_text()
    if os.name == "posix":
        assert stat.S_IMODE(env_file.stat().st_mode) == 0o600


def test_a_dotenv_token_is_read_and_a_malformed_one_is_refused_without_echo(tmp_path):
    env_file = tmp_path / "server.env"
    env_file.write_text(f"HF_TOKEN={DUMMY}\n")
    result = run_secrets(tmp_path, ["--check", "--env-file", str(env_file)])
    assert result.returncode == 0, result.stderr
    assert "set (server.env)" in result.stdout and DUMMY not in result.stdout
    env_file.write_text("HF_TOKEN=not-a-token-secretish\n")
    result = run_secrets(tmp_path, ["--check", "--env-file", str(env_file)])
    assert result.returncode == 2
    assert "secretish" not in result.stdout + result.stderr


def test_a_tracked_or_unignored_env_file_is_refused(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / ".gitignore").write_text(".env\n")
    loose = repo / "loose.env"
    loose.write_text("")
    result = run_secrets(tmp_path, ["--check", "--env-file", str(loose)])
    assert result.returncode == 2 and "not git-ignored" in result.stderr
    tracked = repo / ".env"
    tracked.write_text("")
    subprocess.run(["git", "-C", str(repo), "add", "-f", ".env"], check=True)
    result = run_secrets(tmp_path, ["--check", "--env-file", str(tracked)])
    assert result.returncode == 2 and "tracked by git" in result.stderr
