# Copyright (c) 2026 Seoul National University
# SPDX-License-Identifier: MIT

"""sam3d-scan-models: fetch the two gated checkpoints with the user's existing Hugging Face login.

    facebook/sam3             -> the Hugging Face cache (transformers reads it by repo id)
    facebook/sam-3d-objects   -> --checkpoints (default <sam3d-root>/checkpoints/hf): pipeline.yaml and the weights

Access to both repositories is gated: request it on the model pages first (Meta approves each request; the weights
are licensed under Meta's SAM License, which this project does not redistribute). The token is read by
huggingface_hub from HF_TOKEN, HF_TOKEN_PATH or $HF_HOME/token (default ~/.cache/huggingface/token). This script never reads, copies or prints it.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

from .config import SAM3_REPO_ID, SAM3D_OBJECTS_REPO_ID


def has_login() -> bool:
    """True when a token exists. It is only tested for presence, never read."""
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    explicit = os.environ.get("HF_TOKEN_PATH")
    if explicit:
        return Path(explicit).is_file()
    home = Path(os.environ.get("HF_HOME") or Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "huggingface")
    return (home / "token").is_file()


def checkpoints_present(checkpoints: Path) -> bool:
    return (checkpoints / "pipeline.yaml").is_file()


def _explain(repo_id: str, error: Exception) -> str:
    name = type(error).__name__
    if name in ("GatedRepoError", "RepositoryNotFoundError") or "403" in str(error) or "401" in str(error):
        return (f"{repo_id}: access refused ({name}). Request access at https://huggingface.co/{repo_id} "
                f"with the account that owns the token, then run this again.")
    return f"{repo_id}: {name}: {error}"


def download(checkpoints: Path, which: str, dry_run: bool) -> int:
    if which in ("all", "sam3d-objects") and checkpoints.is_dir() and any(checkpoints.iterdir()) \
            and not checkpoints_present(checkpoints):
        print(f"error: {checkpoints} is not empty and holds no pipeline.yaml: refusing to replace it. Point "
              "--checkpoints at an empty or dedicated directory.", file=sys.stderr)
        return 2
    if not has_login():
        print("error: no Hugging Face login. Run `hf auth login` (it writes ~/.cache/huggingface/token).",
              file=sys.stderr)
        return 2
    if dry_run:
        print(f"would download: {which} (sam3 -> HF cache, sam3d-objects -> {checkpoints})")
        return 0

    from huggingface_hub import snapshot_download  # imported late: --help works without it

    failures = 0
    if which in ("all", "sam3"):
        try:
            path = snapshot_download(repo_id=SAM3_REPO_ID)
            print(f"{SAM3_REPO_ID}: cached at {path}")
        except Exception as e:
            print(_explain(SAM3_REPO_ID, e), file=sys.stderr)
            failures += 1
    if which in ("all", "sam3d-objects"):
        final = checkpoints
        final.parent.mkdir(parents=True, exist_ok=True)
        # Scratch directories of this run only (never a name a user may already have), beside the target so the
        # final move is a rename on the same file system.
        staging = Path(tempfile.mkdtemp(dir=final.parent, prefix=f".{final.name}-download-"))
        old = None
        try:
            snapshot_download(repo_id=SAM3D_OBJECTS_REPO_ID, local_dir=str(staging), max_workers=1)
            # The repository stores the files under checkpoints/; the server reads <checkpoints>/pipeline.yaml.
            inner = staging / "checkpoints"
            new = inner if inner.is_dir() else staging
            # Replace a previous install (checked above) only after the new one is complete; restore it on failure.
            if final.exists():
                old = Path(tempfile.mkdtemp(dir=final.parent, prefix=f".{final.name}-old-"))
                old.rmdir()
                final.rename(old)
            try:
                if new is staging:
                    staging.rename(final)
                else:
                    shutil.move(str(new), str(final))
            except Exception:
                if old is not None and not final.exists():
                    old.rename(final)
                    old = None
                raise
            print(f"{SAM3D_OBJECTS_REPO_ID}: installed at {final}")
        except Exception as e:
            print(_explain(SAM3D_OBJECTS_REPO_ID, e), file=sys.stderr)
            failures += 1
        finally:
            if old is not None:
                shutil.rmtree(old, ignore_errors=True)
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
    return 1 if failures else 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="sam3d-scan-models", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sam3d-root", type=Path, default=os.environ.get("SAM3D_OBJECTS_ROOT"),
                   help="checkout of facebookresearch/sam-3d-objects (env SAM3D_OBJECTS_ROOT)")
    p.add_argument("--checkpoints", type=Path, default=os.environ.get("SAM3D_OBJECTS_CHECKPOINTS"),
                   help="where the SAM 3D Objects checkpoints go (env SAM3D_OBJECTS_CHECKPOINTS; "
                        "default <sam3d-root>/checkpoints/hf)")
    p.add_argument("--only", choices=["all", "sam3", "sam3d-objects"], default="all")
    p.add_argument("--check", action="store_true", help="report what is present, download nothing")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    checkpoints = (Path(a.checkpoints).expanduser() if a.checkpoints
                   else Path(a.sam3d_root).expanduser() / "checkpoints" / "hf" if a.sam3d_root else None)
    if a.check:
        print(f"huggingface login: {'present' if has_login() else 'MISSING'}")
        if checkpoints is not None:
            print(f"sam3d-objects checkpoints: {'present' if checkpoints_present(checkpoints) else 'MISSING'}")
        return 0
    if a.only != "sam3" and checkpoints is None:
        print("error: --checkpoints or --sam3d-root (env SAM3D_OBJECTS_CHECKPOINTS / SAM3D_OBJECTS_ROOT) is required "
              "to place the SAM 3D Objects checkpoints", file=sys.stderr)
        return 2
    return download(checkpoints or Path("."), a.only, a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
