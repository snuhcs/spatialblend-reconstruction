# syntax=docker/dockerfile:1
#
# Two targets:
#   stub  CPU only, no models: the deterministic stand-in for client and network testing (small, built in CI).
#   gpu   the real server: CUDA 12.1, torch 2.5.1, SAM 3D Objects at the pinned commit. Contains NO weights; mount a
#         volume at /models and fill it once with sam3d-scan-models (see README.md).
#
# Build from the repository root (the proto lives in the top-level proto/ directory):
#   docker build -f server/Dockerfile --target stub -t sam3d-scan-server:stub .
#   docker build -f server/Dockerfile --target gpu  -t sam3d-scan-server:gpu  .

# ---------------------------------------------------------------------------------------------------------------------
FROM python:3.11-slim AS stub

WORKDIR /src/server
COPY proto /src/proto
COPY server/pyproject.toml server/README.md server/LICENSE server/NOTICE ./
COPY server/src src
RUN pip install --no-cache-dir . && useradd --create-home scan
USER scan
ENV SAM3D_SCAN_HOST=0.0.0.0 \
    SAM3D_SCAN_PORT=9800
EXPOSE 9800
HEALTHCHECK --interval=10s --timeout=5s --start-period=10s CMD ["sam3d-scan-server", "health", "--server", "127.0.0.1:9800"]
ENTRYPOINT ["sam3d-scan-server", "--stub"]

# ---------------------------------------------------------------------------------------------------------------------
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS gpu

# Compute capabilities the source-built extensions (pytorch3d, gsplat, ...) are compiled for. 8.0 A100, 8.6 RTX 30xx /
# A6000, 8.9 RTX 40xx / L40, 9.0 H100. Narrow it to your GPU to shorten the build.
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
ARG MINIFORGE_VERSION=24.11.3-0
ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    CUDA_HOME=/usr/local/cuda \
    PATH=/opt/conda/bin:${PATH}

RUN apt-get update \
 && apt-get install -y --no-install-recommends git curl ca-certificates build-essential libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL -o /tmp/miniforge.sh \
      "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh" \
 && bash /tmp/miniforge.sh -b -p /opt/conda && rm /tmp/miniforge.sh \
 && conda install -y python=3.11 cmake ninja && conda clean -afy

WORKDIR /src/server
COPY proto /src/proto
COPY server/pyproject.toml server/README.md server/LICENSE server/NOTICE ./
COPY server/src src
COPY server/scripts scripts
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 \
      torch==2.5.1 torchvision==0.20.1 \
 && pip install --no-cache-dir '.[gpu]'
RUN SAM3D_PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121 \
    ./scripts/install_sam3d_objects.sh /opt/sam-3d-objects \
 && pip cache purge

# Weights live on the /models volume, never in the image.
ENV SAM3D_OBJECTS_ROOT=/opt/sam-3d-objects \
    SAM3D_OBJECTS_CHECKPOINTS=/models/sam-3d-objects \
    HF_HOME=/models/hf \
    SAM3D_SCAN_HOST=0.0.0.0 \
    SAM3D_SCAN_PORT=9800
VOLUME /models
EXPOSE 9800
# Loading takes a few minutes; the health service says NOT_SERVING until both models are loaded.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s CMD ["sam3d-scan-server", "health", "--server", "127.0.0.1:9800"]
ENTRYPOINT ["sam3d-scan-server"]
