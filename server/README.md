# sam3d-scan-server

A gRPC server that turns a photo and a few clicks into a 3D object. It is the server half of the object scan in
[SpatialBlend](https://github.com/snuhcs/spatialblend)'s Unity app: the headset photographs an
object, the user marks it, and this server segments it with
[SAM 3](https://huggingface.co/facebook/sam3) and reconstructs a textured mesh with
[SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects).

| Call | In | Out | Model | Typical time (RTX A6000) |
|------|----|-----|-------|------|
| `Segment` | image, clicks and/or one box | `uint8` mask (`0` background, `1` object) | SAM 3 (`Sam3Tracker*` in Transformers) | 1 to 2 s |
| `Reconstruct` | image, mask | `.glb` mesh + pose JSON | SAM 3D Objects | about 35 s |

## The API

The contract is the repository's [`proto/distributed.proto`](../proto/distributed.proto), package `spatialblend.scan.v1`, service
`ReconstructionService` (method paths `/spatialblend.scan.v1.ReconstructionService/Segment` and `.../Reconstruct`).

* **Prompts.** `SegmentRequest.points` are pixel coordinates; `SegPoint.z` is the label: `0` or `1` a positive click,
  `-1` a negative click, `2` and `3` the top-left and bottom-right corners of one box. A request needs a box or at least
  one positive click. The reply's `message` contains `box:1` when the server used the box.
* **Masks** are raw bytes, `mask_height * mask_width`, row-major, the size of the input image.
* **Reconstruct** returns a binary glTF and a JSON document with the object's pose in the camera frame of the image
  (`obj_2_pt3d_transform`: `scale`, `rotationQuaternion`, `translation`, `matrix`).
* **Readiness** is the standard [`grpc.health.v1`](https://github.com/grpc/grpc/blob/master/doc/health-checking.md)
  service: `NOT_SERVING` while the models load (a few minutes) and if a model failed to load, `SERVING` after.
* Messages may be up to 100 MB each way (`SAM3D_SCAN_MAX_MESSAGE_MB`).

The field numbers are pinned by `tests/test_wire_contract.py`. The C++ client in [`../client`](../client) compiles the
same file, so the two cannot drift; another client that keeps its own copy can check it with
`python -m sam3d_scan_server.contract path/to/its/distributed.proto`.

## The client

The app talks to the server through `libspatialblend_session`, the C library in [`../client`](../client) (gRPC client,
plaintext or TLS) that the Unity app ([snuhcs/spatialblend-unity](https://github.com/snuhcs/spatialblend-unity)) loads as a
native plugin; its prebuilt binaries are this repository's GitHub releases. The app has no scan server by default, so
scanning is off until you set your server's address
([docs/local-session-config.md](https://github.com/snuhcs/spatialblend-unity/blob/main/docs/local-session-config.md)
in the Unity repository).

Any gRPC client generated from the proto works too; `tools/probe_stub.py` and `tools/probe_distributed.py` are small
Python examples.

## Requirements

* Linux x86-64 with an NVIDIA GPU with **at least 32 GB** of memory (the server holds both models: 15 to 26 GB resident),
  a recent driver (CUDA 12.1 wheels). Tested on an RTX A6000 (48 GB).
* About 25 GB of disk for the environment and 20 GB for the weights.
* Access to the two gated model repositories (below).

The tests and the stub need neither a GPU nor the weights.

## Model weights

The weights are **not** part of this repository and are never redistributed by it. Meta distributes them under the
[SAM License](https://github.com/facebookresearch/sam-3d-objects/blob/main/LICENSE) through gated Hugging Face
repositories:

1. Request access on <https://huggingface.co/facebook/sam3> and <https://huggingface.co/facebook/sam-3d-objects> with
   your Hugging Face account and wait for approval.
2. Log in once on the server host: `hf auth login` (or export `HF_TOKEN`). The tools here only test that a login exists;
   they never read, copy or print the token.
3. Download: `sam3d-scan-models --sam3d-root ~/sam-3d-objects` puts SAM 3 into the Hugging Face cache and the SAM 3D
   Objects checkpoints into `~/sam-3d-objects/checkpoints/hf` (or `--checkpoints DIR`). `sam3d-scan-models --check`
   reports what is present.

By using the models you accept Meta's SAM License, including its acknowledgement requirement for publications and its
trade-control restrictions. SAM 3D Objects also downloads MoGe (`Ruicheng/moge-vitl`, MIT) on first use.

## Install and run (one script)

`scripts/setup_server.sh` does all of the below from a fresh clone in a `uv` (or `python3.11 -m venv`) environment at
`server/.venv`: it checks the Hugging Face token (from `HF_TOKEN`, `server/.env` or a hidden prompt of
`scripts/configure_secrets.py`), installs the GPU stack and SAM 3D Objects, downloads the weights and, with `--run`,
starts the server. `--cpu` stops after the CPU tests and a stub probe (any OS); `--skip-models` reuses downloaded
weights; `--venv active` installs into an environment you activated yourself. See the repository README's
"Scan server" section.

## Install and run (conda)

```bash
cd server
conda env create -f environment.yml && conda activate sam3d-scan
./scripts/install_sam3d_objects.sh ~/sam-3d-objects       # pinned commit of facebookresearch/sam-3d-objects
sam3d-scan-models --sam3d-root ~/sam-3d-objects
sam3d-scan-server --gpu 0 --sam3d-root ~/sam-3d-objects   # 127.0.0.1:9800
sam3d-scan-server health --server localhost:9800 --wait 900
```

`install_sam3d_objects.sh` builds pytorch3d, gsplat and other CUDA extensions; set `CUDA_HOME` to a CUDA 12.1 toolkit
(for example a conda `cuda-nvcc` 12.1 install) if the system one differs. `SIGINT`/`SIGTERM` stop the server within
five seconds.

## Run with Docker

```bash
docker build -f server/Dockerfile --target gpu -t sam3d-scan-server:gpu .   # from the repository root; long
docker volume create sam3d-models
docker run --rm -e HF_TOKEN -v sam3d-models:/models --entrypoint sam3d-scan-models sam3d-scan-server:gpu
docker run -d --gpus '"device=0"' -p 127.0.0.1:9800:9800 -v sam3d-models:/models sam3d-scan-server:gpu
```

The image holds the code only; the weights stay on the `/models` volume. Pass the token with `-e HF_TOKEN`
as above rather than running `hf auth login` in the container, which would store it on the shared volume.
Set `--build-arg TORCH_CUDA_ARCH_LIST=8.6` (your GPU's compute capability) to shorten the build. The `stub` target
(`docker build -f server/Dockerfile --target stub -t sam3d-scan-server:stub .`) is a CPU image of about 370 MB that answers the same API with
deterministic fake results, for client development and network tests.

## Configuration

Every setting is a flag and an environment variable; the flag wins. Nothing defaults to a particular machine.

| Variable | Flag | Default |
|---|---|---|
| `SAM3D_SCAN_HOST` | `--host` | `127.0.0.1` (loopback only) |
| `SAM3D_SCAN_PORT` | `--port` | `9800` |
| `SAM3D_SCAN_GPUS` | `--gpu N [N ...]` | `0`. Physical indices; the server sets `CUDA_VISIBLE_DEVICES`. Several GPUs give several workers per model. |
| `SAM3D_OBJECTS_ROOT` | `--sam3d-root` | required (not for `--stub`) |
| `SAM3D_OBJECTS_CHECKPOINTS` | `--checkpoints` | `<sam3d-root>/checkpoints/hf` |
| `TORCH_CUDA_ARCH_LIST` | `--cuda-arch` | detected with `nvidia-smi` |
| `SAM3D_SCAN_DEBUG_DIR` | `--debug-dir` | off. When set, one mask overlay PNG per Segment request is written there. |
| `SAM3D_SCAN_LOG_LEVEL` | `--log-level` | `INFO` |
| `SAM3D_SCAN_MAX_MESSAGE_MB` | | `100` |
| `SAM3D_SCAN_REQUEST_TIMEOUT_S` | | `600` |
| `HF_HOME`, `HF_TOKEN` | | Hugging Face defaults |

The server sets `CUDA_VISIBLE_DEVICES` itself from `--gpu`/`SAM3D_SCAN_GPUS` (default `0`), overriding any value in
the environment, and defaults `CUDA_DEVICE_ORDER` to `PCI_BUS_ID` so the indices match `nvidia-smi`.

To listen on every interface, set `SAM3D_SCAN_HOST=0.0.0.0` (or `[::]`); see the next section first.

## Exposing the server

The server has **no authentication and no TLS**. It binds loopback by default. To reach it from a headset, put it
behind a TLS-terminating reverse proxy that forwards gRPC (for example nginx `grpc_pass grpc://127.0.0.1:9800;` in a
`location /spatialblend.scan.v1.` block, with read timeouts above 300 s and a body limit of about 32 MB), and restrict
who can reach it (VPN, allowlist or an auth layer in the proxy). Anyone who can reach the port can use the GPU.

Requests carry the user's photos. The server keeps nothing on disk unless `SAM3D_SCAN_DEBUG_DIR` is set; its logs
record the client id, image size and prompt counts.

## Tests

```bash
cd server && pip install -e '.[test]' && pytest
```

CPU only: no GPU, no torch, no weights. They cover the prompt parser, the wire contract, configuration, the servicer
with the models replaced by fakes over real gRPC, health and readiness, fail-fast pools, the model-load lock, clean
shutdown, and the stub. Two tests run against a real server when `SAM3D_SCAN_REAL_SERVER=host:port` is set. CI runs
the tests on Python 3.10 to 3.12 and builds and probes the stub image.

## Credits

Designed and implemented by Dongho Han (Seoul National University); packaging, the health service, the stub, the
`spatialblend.scan.v1` API names and box and negative prompts by Jingyu Lee.

## Licence

MIT, see [LICENSE](LICENSE). The models it runs are not covered by that licence: see [NOTICE](NOTICE) and
[THIRD_PARTY.md](THIRD_PARTY.md).
