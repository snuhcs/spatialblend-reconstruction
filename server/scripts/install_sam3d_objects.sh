#!/usr/bin/env bash
# Copyright (c) 2026 Seoul National University
# SPDX-License-Identifier: MIT

# Clone facebookresearch/sam-3d-objects at the pinned commit and install it into the active environment.
# Usage: scripts/install_sam3d_objects.sh <checkout-dir>
# The checkout dir is what --sam3d-root / SAM3D_OBJECTS_ROOT points at.
set -euo pipefail

PIN=f91db411c50efee93d8db7aeb323885650f6f722
DEST=${1:?usage: install_sam3d_objects.sh <checkout-dir>}

if [ ! -d "$DEST/.git" ]; then
  git clone https://github.com/facebookresearch/sam-3d-objects "$DEST"
fi
git -C "$DEST" fetch --quiet origin "$PIN" || true
git -C "$DEST" checkout --quiet "$PIN"

# Override both when a host cannot resolve pypi.ngc.nvidia.com (pip retries every package against a dead index):
#   SAM3D_PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121"
export PIP_EXTRA_INDEX_URL="${SAM3D_PIP_EXTRA_INDEX_URL:-https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121}"
# nvidia-pyindex is a build-time helper that points pip at pypi.ngc.nvidia.com; nothing imports it. On a host that
# cannot resolve that name its build fails and aborts the whole install, so a no-op stand-in satisfies the pin.
if ! getent hosts pypi.ngc.nvidia.com >/dev/null 2>&1; then
  echo "pypi.ngc.nvidia.com does not resolve: installing a no-op nvidia-pyindex stand-in"
  STUB=$(mktemp -d)
  mkdir -p "$STUB/nvidia_pyindex"
  printf '' > "$STUB/nvidia_pyindex/__init__.py"
  printf '[build-system]\nrequires=["setuptools"]\nbuild-backend="setuptools.build_meta"\n[project]\nname="nvidia-pyindex"\nversion="1.0.9"\n' > "$STUB/pyproject.toml"
  pip install "$STUB"
fi
( cd "$DEST"
  pip install -e '.[dev]'
  # pytorch3d, flash-attn and gsplat compile against the installed torch, so they must build without pip's
  # isolated environment (which has no torch). Their build tools go in first.
  pip install wheel packaging ninja setuptools hatchling hatch-requirements-txt editables
  # Source builds need a CUDA 12.1 toolkit that matches torch (cu121). Point CUDA_HOME at one (a conda
  # cuda-nvcc 12.1 install works); a mismatching system nvcc makes torch's extension build refuse.
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  export MAX_JOBS="${MAX_JOBS:-8}"
  # pytorch3d: official wheel, then the community wheel index, then a source build.
  P3D_WHEELS="https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py311_cu121_pyt251/download.html"
  pip install --no-index --find-links "$P3D_WHEELS" pytorch3d \
    || pip install --no-deps "pytorch3d==0.7.8+pt2.5.1cu121" --extra-index-url https://miropsota.github.io/torch_packages_builder \
    || pip install --no-build-isolation 'pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47'
  # flash-attn fetches a prebuilt wheel when one matches and builds otherwise. It is an optional speed-up:
  # a failure is reported, not fatal.
  pip install --no-build-isolation flash_attn==2.8.3 || echo "WARNING: flash_attn did not install; sdpa attention is used"
  export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
  pip install --no-build-isolation -e '.[inference]'
  ./patching/hydra             # hydra patch for facebookresearch/hydra#2863
)
# transformers >= 5 for Sam3Tracker*, but later 5.x need torch >= 2.6 and sam3d_objects pins torch 2.5.1: 5.3.0 is the
# tested release. Install it last so the sam3d_objects pins do not downgrade it. lightning (via sam3d_objects) still
# imports pkg_resources, which setuptools 81 removed.
pip install 'transformers==5.3.0' 'setuptools<81'
echo "sam3d-objects installed from $PIN at $DEST"
echo "next: sam3d-scan-models --sam3d-root $DEST"
