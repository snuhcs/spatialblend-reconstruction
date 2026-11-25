# Third-party components

Nothing below is vendored in this repository. The table lists what the server installs or loads at run time and the
licence each is distributed under, so that deployers (and anyone publishing a container image) know their obligations.

## Models and model code (not redistributed here)

| Component | Source | Licence | Notes |
|---|---|---|---|
| SAM 3 weights and processor | `facebook/sam3` on Hugging Face (gated, approval by Meta) | SAM License (Meta) | Loaded by `transformers` (`Sam3TrackerModel`). Tested revision `3c879f39826c281e95690f02c7821c4de09afae7`. |
| SAM 3D Objects code | `facebookresearch/sam-3d-objects` | SAM License (Meta) | Installed by `scripts/install_sam3d_objects.sh` at commit `f91db411c50efee93d8db7aeb323885650f6f722`. |
| SAM 3D Objects checkpoints | `facebook/sam-3d-objects` on Hugging Face (gated, approval by Meta) | SAM License (Meta) | About 13 GB. Tested revision `2e73555018d2741ccd486e56c24fac41155a1dc6`. |
| MoGe ViT-L | `Ruicheng/moge-vitl` on Hugging Face | MIT | Fetched by SAM 3D Objects' pipeline (depth model). |

The SAM License permits use, modification and redistribution, provided a copy of the licence accompanies any
redistribution; it requires acknowledging SAM in publications of research results and forbids uses subject to ITAR or
prohibited by trade controls (named: military or warfare, nuclear, espionage, weapons) and reverse engineering of
the SAM Materials; read the licence itself before deploying. This project nevertheless does not redistribute the weights: Meta distributes them through
gated repositories, and the README tells users how to request access. A container image built from `Dockerfile` (`gpu`
target) contains the SAM 3D Objects code (with its `LICENSE`) but no weights; publishing such an image redistributes
SAM Materials and must keep that licence file.

## Python packages

| Package | Licence | Used for |
|---|---|---|
| grpcio, grpcio-tools, grpcio-health-checking | Apache-2.0 | gRPC server, proto compilation at first use, health service |
| protobuf | BSD-3-Clause | messages |
| numpy | BSD-3-Clause | masks |
| Pillow | MIT-CMU (HPND) | image decoding, debug overlays |
| huggingface_hub | Apache-2.0 | `sam3d-scan-models` downloads |
| torch, torchvision | BSD-3-Clause | GPU inference |
| transformers | Apache-2.0 | SAM 3 tracker model and processor |
| pytorch3d | BSD-3-Clause | transforms (via SAM 3D Objects) |
| kaolin | Apache-2.0 | dependency of SAM 3D Objects |
| gsplat | Apache-2.0 | dependency of SAM 3D Objects |
| flash-attn (optional) | BSD-3-Clause | attention speed-up |
| pytest (tests only) | MIT | tests |

SAM 3D Objects pulls further packages through its own `requirements*.txt`; see that repository.
