# SpatialBlend Session

SpatialBlend Session is a C/C++ client library for remote inference. It connects to
a gRPC inference server (SAM3 segmentation and SAM3D object reconstruction),
sends asynchronous requests and delivers results through callbacks and polled
event handles. It is exposed as a C API (`libspatialblend_session`, all exported
symbols prefixed `spatialblend_`) so that Unity, C# and other native hosts can use
it on Linux, macOS, Windows and Android. It is separate from, and shares no
symbols with, the core library in
[spatialblend-core](https://github.com/snuhcs/spatialblend-core).

The server it talks to is in [../server](../server) of this repository
(SAM 3 segmentation and SAM 3D Objects reconstruction, gRPC package
`spatialblend.scan.v1`). The SpatialBlend Unity app
([snuhcs/spatialblend-unity](https://github.com/snuhcs/spatialblend-unity)) loads this
library as a native plugin for its object scan. Prebuilt libraries are in this
repository's [releases](https://github.com/snuhcs/spatialblend-reconstruction/releases).

## Build and test

Requires Bazel 8.4.2 (`.bazelversion`). Run Bazel in this directory (`client/`);
the build also reads the shared `../proto/distributed.proto`.

```sh
bazel build //session/...
bazel build //session/c:session_c                    # C API library
bazel test //session/c:c_api_test --test_output=errors
bazel build //session/c:libspatialblend_session_c       # tarball with library and headers

python script/build_c_api.py            # per-platform build; see --help (-all, -a, -d, -docker, -o)
```

`build_c_api.py` unpacks each platform's build into `output/<platform>/lib/` (the
shared library) and `output/<platform>/include/spatialblend_session/` (the headers).

`bazel test //session/...` runs the C API test and an offline TLS test (it
generates a self-signed certificate at test time). Two tests are tagged
`manual` because they need a running server:
`//session/inference/distributed:distributed_test`
(`SPATIALBLEND_DISTRIBUTED_ADDR=<host>:<port>`, the scan API) and
`//session/inference/sam3d:sam3d_remote_test` (`SPATIALBLEND_SAM3D_ADDR`, the
legacy one-shot API); pass them with `--test_env`. Their inputs are the drawn
images in `session/inference/sam3d/test_data/` (`script/make_test_data.py`).
For the scan API, `sam3d-scan-server --stub` (`pip install -e ../server`) is
a CPU-only stand-in. On Windows, deep external paths can exceed MAX_PATH; use a short
Bazel output base, as `script/build_c_api.py` does automatically.

## Using the C API

Create a session for a model type, connect, submit a request, then receive the
result through a registered callback or by waiting on the returned handle. The
entry points are declared in [session/c/spatialblend_session.h](session/c/spatialblend_session.h):
`spatialblend_inference_session_create`, `spatialblend_inference_session_connect`,
`spatialblend_inference_segment`, `spatialblend_inference_reconstruct`,
`spatialblend_inference_sam3d`, `spatialblend_inference_request`,
`spatialblend_register_on_inference_complete` and `spatialblend_cleanup_all`. Waiting
on a returned handle uses `spatialblend_try_wait_for_event`, declared in
[session/c/spatialblend_session_utility.h](session/c/spatialblend_session_utility.h).

The session type picks the pipeline (`sb_inference_type_t` in
[session/c/spatialblend_session_type.h](session/c/spatialblend_session_type.h)):
`SB_INFERENCE_TYPE_SAM3_SEG` (segmentation only) and `SB_INFERENCE_TYPE_SAM3D_OBJ`
(reconstruction only) use the split server in `session/inference/distributed/`;
`SB_INFERENCE_TYPE_SAM3D` is the legacy full pipeline (`session/inference/sam3d/`).
A scan uses two sessions: `spatialblend_inference_segment` on a `SAM3_SEG` session
returns a mask, then `spatialblend_inference_reconstruct` on a `SAM3D_OBJ` session
turns the image and mask into `<output_path>_objmodel.glb` and
`<output_path>_objinfo.json`.

Requests return immediately with an event handle; the library copies the input
buffers before returning, so the caller may release them. Completion callbacks
run on a library thread, not the caller's: a host such as Unity must hand the
result to its main thread itself. Threading and lifetime details are in
[CONTRIBUTING.md](CONTRIBUTING.md#architecture).

### From Unity (C#)

Declare the entry points with `[DllImport]`. Android, macOS and Linux load
`"spatialblend_session"` (the loader adds the `lib` prefix and the extension);
Windows loads the file name as given, so use `"libspatialblend_session"` there
(`#if UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN`). Release files go to
`Assets/Plugins/Android/arm64-v8a/libspatialblend_session.so`,
`Assets/Plugins/OSX/libspatialblend_session.dylib` and
`Assets/Plugins/Windows/libspatialblend_session.dll`; for Linux, install
`libspatialblend_session-linux-x64.so` as
`Assets/Plugins/x86_64/libspatialblend_session.so` (import settings: Editor and
Standalone, Linux x86_64). Add the `android.permission.INTERNET` permission on Android.

Usage rules:

- **Lifecycle order.** `spatialblend_inference_session_create` →
  `spatialblend_register_on_inference_complete` → `spatialblend_inference_session_connect`
  (or `_connect_tls`) → requests (`spatialblend_inference_segment`,
  `spatialblend_inference_reconstruct`) → `spatialblend_inference_session_destroy`, or
  `spatialblend_cleanup_all` to release every session and log listener.
- **Callbacks.** A callback runs on a library thread. Under IL2CPP it must be a
  `static` method marked `[AOT.MonoPInvokeCallback(typeof(...))]`; keep the
  delegate alive for as long as it is registered and hand results to the main
  thread before touching Unity APIs.
- **Buffers.** Pin a byte array only for the duration of the call; the library
  copies it before returning.
- **Errors.** A request returns an event handle. `spatialblend_try_wait_for_event(handle,
  timeout_ms)` returns true once the event is signalled; on failure,
  `spatialblend_get_error_message(handle)` returns the error text (for example from a
  failed connect). Results of an inference arrive through the callback.
- **Loading the model.** The `_objmodel.glb` file is binary glTF; a glTF loader such
  as glTFast (`com.unity.cloud.gltfast`) loads it.

The Unity app's `Assets/SpatialBlend/Framework/Internal/SessionLibrary.cs` is a complete example.

Adding a model: implement `InferenceClientBase` under `session/inference/<model>/`,
add a type to `sb_inference_type_t`, and add a case to the factory in
`session/c/spatialblend_session.cc`.

## Golden pipeline

`script/golden_pipeline/golden_pipeline.py` drives the built shared library
through segmentation and reconstruction against a live scan server and compares
the output files with stored goldens (exit 0 ok, 1 drift, 2 setup error):

```sh
python script/golden_pipeline/golden_pipeline.py check --lib <libspatialblend_session> --addr <host:port> --golden-dir <dir>
python script/golden_pipeline/golden_pipeline.py record ...     # write goldens and calibration.json
python script/golden_pipeline/golden_pipeline.py selftest       # harness check against the bundled stub server
```

Goldens are model outputs and are kept outside this repository; only
`thresholds.json` is committed. `--lib`, `--addr` and `--golden-dir` default to
`SPATIALBLEND_SESSION_LIB`, `SPATIALBLEND_SAM3D_ADDR` and `SPATIALBLEND_GOLDEN_DIR`.

## Repository layout

| Path | Responsibility |
|---|---|
| [session/c/](session/c/) | C API, export list and version script |
| [session/inference/](session/inference/) | `InferenceClientBase`; `sam3d/` (legacy full pipeline) and `distributed/` (SAM3 segmentation, SAM3D object clients) with their protos |
| [session/network/](session/network/) | Async gRPC client base and proto utilities |
| [session/common/](session/common/), [session/util/](session/util/) | Shared types and helpers |
| [script/](script/) | `build_c_api.py`, export audit, golden pipeline |
| [third_party/](third_party/) | Build files for the Android NDK and stb |

## The scan protocol

The scan API is the repository's top-level
[`proto/distributed.proto`](../proto/distributed.proto), the single source of
truth for this library and the server. Bazel reads it through the `@scan_proto`
repository (`MODULE.bazel`) and copies it into
`session/inference/distributed/proto/` at build time, so the generated headers
keep their include paths; there is no checked-in copy.

`SegPoint.z` carries the prompt label (`0`/`1` positive click, `-1` negative,
`2`/`3` box corners); see the comments in the proto.
`session/inference/sam3d/proto/sam3d.proto` is the older one-shot API
(`OneShotReconstructionService`), which the current server does not serve.

## CI and binaries

`../.github/workflows/ci.yml` (GitHub-hosted) runs the tests on Linux, builds the
library, audits that only `spatialblend_*` symbols are exported, and runs the
golden-pipeline self-test against the bundled stub server.
`../.github/workflows/client-release.yml` builds Android arm64, macOS arm64,
Windows x64 and Linux x64, audits each, and (on a `v*` tag or a manual run)
creates a draft GitHub Release with `libspatialblend_session.so` (Android arm64),
`libspatialblend_session.dylib`, `libspatialblend_session.dll`,
`libspatialblend_session-linux-x64.so`, the headers, `SHA256SUMS` and
`THIRD_PARTY_NOTICES-native.txt`. The Unity app's setup script downloads
those files.

## Licence

MIT, see [../LICENSE](../LICENSE). Third-party code: [THIRD_PARTY.md](THIRD_PARTY.md);
the binaries ship with [THIRD_PARTY_NOTICES-native.txt](THIRD_PARTY_NOTICES-native.txt).

## Documentation

Architecture, threading and contributor notes: [CONTRIBUTING.md](CONTRIBUTING.md).

## Credits

Designed and implemented by Dongho Han (Seoul National University); TLS
transport, the scan API names, CI and the golden pipeline by Jingyu Lee.
