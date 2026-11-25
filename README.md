# SpatialBlend Reconstruction

Object scanning for [SpatialBlend](https://github.com/snuhcs/spatialblend) (UIST '26): the headset photographs an
object, the user marks it with clicks or a box, and a GPU server segments it with
[SAM 3](https://huggingface.co/facebook/sam3) and reconstructs a 3D mesh with
[SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects). This repository holds that server and
`libspatialblend_session`, the client library the Unity app
[snuhcs/spatialblend-unity](https://github.com/snuhcs/spatialblend-unity) loads as a native plugin.

[Project page](https://github.com/snuhcs/spatialblend) · [Paper](https://doi.org/10.1145/3830398.3830690) ·
[Video](https://www.youtube.com/watch?v=3TWBkZOWY_4) · [Unity app](https://github.com/snuhcs/spatialblend-unity) ·
[Native core](https://github.com/snuhcs/spatialblend-core)

| Directory | Contents |
|---|---|
| [`proto/`](proto) | The gRPC API (`spatialblend.scan.v1`: `Segment`, `Reconstruct`), compiled by both sides |
| [`server/`](server) | `sam3d-scan-server`, the Python gRPC scan server |
| [`client/`](client) | `libspatialblend_session`, the C API and C++ gRPC client |

## Client library

**Prebuilt:** the [v1.0.0 release](https://github.com/snuhcs/spatialblend-reconstruction/releases/tag/v1.0.0) has
`libspatialblend_session` for Android arm64, macOS arm64, Windows x64 and Linux x64, with the headers and `SHA256SUMS`.
The Unity app's setup script downloads it for you.

**From source** (macOS, Linux or Windows): install [Bazelisk](https://github.com/bazelbuild/bazelisk) as `bazel` and a
C++17 toolchain; for Android also NDK r27d in `ANDROID_NDK_HOME`.

```bash
python3 client/script/build_from_scratch.py --test               # this host: tests and build -> client/output/<platform>/
python3 client/script/build_from_scratch.py --platform android   # Android arm64
```

Details, the C API and its use from Unity: [client/README.md](client/README.md).

## Scan server

Requires Linux x86-64 with an NVIDIA GPU of at least 32 GB, a CUDA 12.1 toolkit, git, and `uv` or Python 3.11.

**Model weights are not included.** Meta distributes SAM 3 and SAM 3D Objects under the SAM License through gated
Hugging Face repositories:

1. Request access to [facebook/sam3](https://huggingface.co/facebook/sam3) and
   [facebook/sam-3d-objects](https://huggingface.co/facebook/sam-3d-objects), and create a read token at
   <https://huggingface.co/settings/tokens>.
2. Provide the token: `export HF_TOKEN=...`, or run `python3 server/scripts/configure_secrets.py`, which stores it in
   the git-ignored `server/.env`.
3. Install everything (environment, SAM 3D Objects, weights) and start the server on `127.0.0.1:9800`:

   ```bash
   server/scripts/setup_server.sh --run
   ```

Without a GPU (any OS), `server/scripts/setup_server.sh --cpu` installs the CPU part, runs the tests and probes the stub
server, a CPU stand-in that answers the same API with fake results (no token, no weights).

The server has no authentication or TLS of its own; put it behind a TLS proxy before a headset reaches it. Conda and
Docker installs, configuration and exposing the server safely: [server/README.md](server/README.md).

## License

MIT, Copyright (c) 2026 Seoul National University ([LICENSE](LICENSE), [NOTICE](NOTICE)). The models the server runs are
not covered: see [server/THIRD_PARTY.md](server/THIRD_PARTY.md); third-party code in the client:
[client/THIRD_PARTY.md](client/THIRD_PARTY.md).

## Citation

Please cite the SpatialBlend paper ([doi:10.1145/3830398.3830690](https://doi.org/10.1145/3830398.3830690)); the BibTeX
is on the [project page](https://github.com/snuhcs/spatialblend#citation).

## Contact

Jingyu Lee (<jingyu.lee@hcs.snu.ac.kr>), Human-Centered Computer Systems Lab, Seoul National University.

## Contributors

- Jingyu Lee (Seoul National University)
- Dongho Han (Seoul National University)
- Jinmyeong Kim (Seoul National University)
