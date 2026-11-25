#!/usr/bin/env python3
# Copyright (c) 2026 Seoul National University
# SPDX-License-Identifier: MIT

"""Record the scan server's own values in server/.env (git-ignored, mode 600), or check them.

    python3 scripts/configure_secrets.py                    # ask for what is missing (hidden prompt for the token)
    python3 scripts/configure_secrets.py --non-interactive  # never prompt; take values from the environment only (CI)
    python3 scripts/configure_secrets.py --save             # also record values found in the environment in server/.env
    python3 scripts/configure_secrets.py --check            # list what is set or missing; change nothing

The only secret is the Hugging Face token (HF_TOKEN). It is needed once, to download the gated SAM 3 and SAM 3D Objects
weights (sam3d-scan-models); the running server does not use it. The server itself has no authentication and no TLS:
put a TLS proxy and an access restriction in front of it (README.md, "Exposing the server"); the proxy's certificate
and keys belong to the proxy, not here. The other keys are plain settings with defaults (README.md, "Configuration").

Each value comes from, in order: the environment, server/.env, a prompt. A Hugging Face login that already exists
(`hf auth login`, HF_TOKEN_PATH) counts for HF_TOKEN and is never read. No value is ever printed: --check reports only
where a value comes from. Values are written only to server/.env, created with mode 600: a prompted value always, a value
from the environment only with --save (so an exported HF_TOKEN is not copied to disk unless asked). The file must not be
tracked by git, and inside a git checkout it must be git-ignored (server/.env is). Template: server/.env.example.

Exit status: 0 all required values are present; 2 a required value is missing, or a usage error.
"""
from __future__ import annotations

import argparse
import getpass
import os
import re
import subprocess
import sys
from pathlib import Path

SERVER = Path(__file__).resolve().parents[1]
ENV_FILE = SERVER / ".env"

# (key, secret, required, default shown in --check, what it is)
KEYS = (
    ("HF_TOKEN", True, True, None,
     "Hugging Face token of an account with access to facebook/sam3 and facebook/sam-3d-objects (model download only)"),
    ("SAM3D_OBJECTS_ROOT", False, False, "~/sam-3d-objects", "checkout of facebookresearch/sam-3d-objects (setup_server.sh clones it)"),
    ("SAM3D_OBJECTS_CHECKPOINTS", False, False, "<SAM3D_OBJECTS_ROOT>/checkpoints/hf", "SAM 3D Objects weights"),
    ("HF_HOME", False, False, "~/.cache/huggingface", "Hugging Face cache (SAM 3 weights)"),
    ("SAM3D_SCAN_HOST", False, False, "127.0.0.1", "listen address (loopback only by default)"),
    ("SAM3D_SCAN_PORT", False, False, "9800", "listen port"),
    ("SAM3D_SCAN_GPUS", False, False, "0", "GPU indices, space separated"),
)
KEY_NAMES = tuple(k[0] for k in KEYS)
LINE = re.compile(r"^\s*(?:export\s+)?([A-Z_][A-Z0-9_]*)\s*=\s*(.*?)\s*$")


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        match = LINE.match(raw)
        if not match:
            continue
        value = match.group(2)
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if value:
            values[match.group(1)] = value
    return values


def env_file_problem(path: Path) -> str | None:
    """Why path must not hold secrets: tracked by git, or inside a git checkout and not git-ignored. None if it may."""
    path = path.expanduser().resolve()
    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(path.parent), *args], capture_output=True, text=True)
    try:
        inside = path.parent.is_dir() and git("rev-parse", "--is-inside-work-tree").returncode == 0
    except OSError:
        return None   # no git: nothing can track it
    if not inside:
        return None
    if git("ls-files", "--error-unmatch", "--", path.name).returncode == 0:
        return f"{path} is tracked by git; untrack it (git rm --cached) and keep secrets out of git"
    if git("check-ignore", "-q", "--", path.name).returncode != 0:
        return f"{path} is inside a git checkout and not git-ignored; use server/.env or add the file to .gitignore"
    return None


def hf_login_exists(env: dict[str, str]) -> bool:
    """A stored Hugging Face login (tested for presence only, as sam3d-scan-models does)."""
    if env.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    if env.get("HF_TOKEN_PATH"):
        return Path(env["HF_TOKEN_PATH"]).expanduser().is_file()
    home = env.get("HF_HOME") or str(Path(env.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "huggingface")
    return (Path(home).expanduser() / "token").is_file()


def write_env_file(path: Path, values: dict[str, str]) -> None:
    """Rewrite path with values (mode 600 from creation). Comments and unknown keys of an existing file are kept."""
    lines: list[str] = []
    done: set[str] = set()
    if path.is_file():
        for raw in path.read_text(encoding="utf-8").splitlines():
            match = LINE.match(raw)
            if match and match.group(1) in values:
                key = match.group(1)
                if key not in done:
                    lines.append(f"{key}={values[key]}")
                    done.add(key)
                continue
            lines.append(raw)
    for key, value in values.items():
        if key not in done:
            lines.append(f"{key}={value}")
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    owner_only(path)


def owner_only(path: Path) -> None:
    """Mode 600; on Windows, where chmod cannot, an ACL for the current user alone (a file on D: inherits Users: read)."""
    os.chmod(path, 0o600)
    if os.name == "nt":
        subprocess.run(["icacls", str(path), "/inheritance:r", "/grant:r", f"{getpass.getuser()}:F"], capture_output=True, check=True)


def validate(key: str, value: str) -> str | None:
    """An error message for a malformed value (never containing the value), or None."""
    if any(c in value for c in "\n\r"):
        return f"{key} contains a line break"
    if key == "HF_TOKEN" and not re.fullmatch(r"hf_[A-Za-z0-9]{20,}", value):
        return "HF_TOKEN does not look like a Hugging Face token (hf_ followed by letters and digits)"
    if key == "SAM3D_SCAN_PORT" and not (value.isdigit() and 0 < int(value) < 65536):
        return "SAM3D_SCAN_PORT must be a port number"
    if key == "SAM3D_SCAN_GPUS" and not re.fullmatch(r"\d+( \d+)*", value):
        return "SAM3D_SCAN_GPUS must be GPU indices separated by spaces"
    return None


def resolve(env: dict[str, str], file_values: dict[str, str]) -> list[tuple[str, str, str | None]]:
    """(key, source, value) for every key. source is env, .env, hf-login, default or missing."""
    rows = []
    for key, _secret, _required, default, _what in KEYS:
        if env.get(key):
            rows.append((key, "env", env[key]))
        elif file_values.get(key):
            rows.append((key, ".env", file_values[key]))
        elif key == "HF_TOKEN" and hf_login_exists({**file_values, **env}):
            rows.append((key, "hf-login", None))
        elif default is not None:
            rows.append((key, "default", None))
        else:
            rows.append((key, "missing", None))
    return rows


def print_rows(rows: list[tuple[str, str, str | None]], env_file: Path) -> None:
    info = {k[0]: k for k in KEYS}
    width = max(len(k) for k in KEY_NAMES)
    print("Scan server values (never printed; only where each comes from)")
    for key, source, _value in rows:
        _k, secret, required, default, what = info[key]
        state = {"env": "set (environment)", ".env": f"set ({env_file.name})", "hf-login": "set (existing Hugging Face login)",
                 "default": f"default {default}", "missing": "MISSING" + (" (required)" if required else "")}[source]
        print(f"  {key.ljust(width)}  {state.ljust(36)}  {what}{' [secret]' if secret else ''}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="list what is set or missing; change nothing")
    parser.add_argument("--non-interactive", action="store_true", help="never prompt; take values from the environment")
    parser.add_argument("--env-file", type=Path, default=ENV_FILE, help=f"the file to read and write (default {ENV_FILE})")
    parser.add_argument("--save", action="store_true",
                        help="also record values found in the environment in the env file (default: only prompted values)")
    parser.add_argument("--no-token", action="store_true",
                        help="the weights are already downloaded: HF_TOKEN is not required")
    args = parser.parse_args(argv)
    args.env_file = args.env_file.expanduser().resolve()
    problem = env_file_problem(args.env_file)
    if problem:
        print(f"configure_secrets: {problem}", file=sys.stderr)
        return 2

    env = dict(os.environ)
    file_values = read_env_file(args.env_file)
    rows = resolve(env, file_values)
    required = {k[0] for k in KEYS if k[2] and not (k[0] == "HF_TOKEN" and args.no_token)}

    for key, source, value in rows:
        if value is not None:
            problem = validate(key, value)
            if problem:
                print(f"configure_secrets: {problem} (from {source})", file=sys.stderr)
                return 2

    if not args.check:
        interactive = not args.non_interactive and sys.stdin.isatty()
        updates: dict[str, str] = {}
        for key, source, value in rows:
            if source == "env" and args.save and file_values.get(key) != value:
                updates[key] = value        # --save: record an environment value so later runs need not repeat it
            elif source == "missing" and key in required and interactive:
                answer = getpass.getpass(f"{key} (hidden; empty to skip): ").strip()
                if answer:
                    problem = validate(key, answer)
                    if problem:
                        print(f"configure_secrets: {problem}", file=sys.stderr)
                        return 2
                    updates[key] = answer
        if updates:
            write_env_file(args.env_file, updates)
            print(f"wrote {', '.join(sorted(updates))} to {args.env_file} (mode 600)")
        elif args.env_file.is_file():
            owner_only(args.env_file)
        file_values = read_env_file(args.env_file)
        rows = resolve(env, file_values)

    print_rows(rows, args.env_file)
    missing = [key for key, source, _v in rows if source == "missing" and key in required]
    if missing:
        print(f"\nmissing: {', '.join(missing)}. Set it in the environment, in {args.env_file} "
              f"(template: {SERVER / '.env.example'}), or run python3 scripts/configure_secrets.py and answer the prompt. "
              "HF_TOKEN: create a read token at https://huggingface.co/settings/tokens after access to "
              "facebook/sam3 and facebook/sam-3d-objects is granted (or pass --no-token when the weights are already present).",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
