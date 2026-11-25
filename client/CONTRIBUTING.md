# Contributing

Notes for working on the library: build, tests, architecture and code style. Use the
`[module] type: description` commit style (for example `[session] fix: ...`).

## Build & Test Commands

```bash
# Build all
bazel build //session/...

# Build C API
bazel build //session/c:session_c

# Run C API test
bazel test //session/c:c_api_test --test_output=errors --nocache_test_results

# Run the remote tests (manual: they require a running server)
bazel test //session/inference/distributed:distributed_test \
  --test_output=streamed \
  --test_env=SPATIALBLEND_DISTRIBUTED_ADDR=<scan-server-host>:9800 \
  --nocache_test_results
bazel test //session/inference/sam3d:sam3d_remote_test \
  --test_output=streamed \
  --test_env=SPATIALBLEND_SAM3D_ADDR=<sam3d-host>:9800 \
  --test_env=SPATIALBLEND_SAM3D_TEST_OUTPUT_DIR="$(realpath ./test_output)" \
  --nocache_test_results
```

**Build configurations:**
- Release (default): `bazel build -c opt`
- Debug: `bazel build --config=debug`

### Windows: MAX_PATH (260 char) Limitation

Windows has a 260-character path limit. Deeply nested external dependencies (e.g. `envoy_api`) can exceed this limit with Bazel's default output base path.

**Build script** (`script/build_c_api.py`): Automatically uses a short output base (`C:/_b/s_w_o`), no action needed.

**Direct `bazel` commands**: Must specify `--output_base` manually:
```bash
bazel --output_base C:/_b/s_w_o test //session/inference/sam3d:sam3d_remote_test \
  --test_output=streamed \
  --test_env=SPATIALBLEND_SAM3D_ADDR=<sam3d-host>:9800 \
  --test_env=SPATIALBLEND_SAM3D_TEST_OUTPUT_DIR="$(realpath ./test_output)" \
  --nocache_test_results
```

**Permanent fix**: Add to user-level `~/.bazelrc` (not project `.bazelrc` — startup options don't support platform configs):
```
startup --output_user_root=C:/_b
```

## Architecture

### Inference clients

```
session/network/grpc/grpc_client_base.h   IGrpcClient (enable_shared_from_this; pending-call tracking)
session/inference/inference_client.h      InferenceClientBase (connection + async CompletionQueue thread)
  session/inference/sam3d/                  Sam3dClient     (legacy one-shot ProcessImage)
  session/inference/distributed/            Sam3SegClient   (Segment), Sam3dObjClient (Reconstruct)
```

- `InferenceClientBase` merges gRPC connection management and the inference interface.
  Concrete: `Connect()`, `Disconnect()`, `IsConnected()`, `HandleRpcs()` (the CompletionQueue loop on
  its own thread). Pure virtual: `ConnectToAddress()`, `Process()`; optional hook `OnConnect()` (create the stub).
- Each RPC is a `GrpcClientCallBase` tag that registers itself as a pending call and holds a `weak_ptr` to its
  client, so a tag finishing after the client is gone does nothing. `Disconnect()` cancels the pending calls,
  shuts the queue down, drains it and joins the queue thread.
- The C API (`session/c/spatialblend_session.cc`) keeps a session registry (handle -> client, type, callback) and a
  `StatusRing` of 1000 event handles. Each request starts the RPC, returns an event handle at once, and finishes on a
  detached thread that waits for the RPC, writes the output files, calls the registered callback and resolves the
  event. That thread holds a `shared_ptr` to the session, so destroying a session mid-request is safe.

### C API Functions

Declared in `session/c/spatialblend_session.h` (and `spatialblend_session_utility.h` for events):

```c
// Session lifecycle
sb_inference_session_t spatialblend_inference_session_create(sb_inference_type_t type);
sb_handle_t spatialblend_inference_session_destroy(session);
sb_handle_t spatialblend_inference_session_connect(session, "host:port");      // plaintext
sb_handle_t spatialblend_inference_session_connect_tls(session, "host:port");  // TLS, cert verified vs host

// Scan API (split server)
sb_handle_t spatialblend_inference_segment(session, client_id, image, size, points, n, mask_path);
sb_handle_t spatialblend_inference_reconstruct(session, client_id, image, size, mask, mask_size, w, h, out_prefix);

// Legacy and generic
sb_handle_t spatialblend_inference_sam3d(session, client_id, image, size, points, n, out_prefix);
sb_handle_t spatialblend_inference_request(session, client_id, &input);

// Callbacks and logs
bool spatialblend_register_on_inference_complete(session, callback);
sb_handle_t spatialblend_register_log_listener(verbosity, callback);
bool spatialblend_unregister_log_listener(callback_id);

// Event system
bool spatialblend_try_wait_for_event(handle, timeout_ms);
const char* spatialblend_get_error_message(handle);

// Cleanup
void spatialblend_cleanup_all();
```

### Adding a New Model

1. Create `session/inference/<model>/<model>_client.h` implementing `InferenceClientBase` (with its proto and `BUILD`)
2. Add the next free value to `sb_inference_type_t` (`spatialblend_session_type.h`; 0 to 3 are taken)
3. Add a case to the factory switch in `spatialblend_inference_session_create()` (`spatialblend_session.cc`)
4. Add a dedicated C entry point only if the generic `spatialblend_inference_request()` does not fit the inputs

### gRPC API

The scan protos use package `spatialblend.scan.v1`:
`ReconstructionService` (`Segment`, `Reconstruct`) and `OneShotReconstructionService`
(`ProcessImage`, legacy). Wire paths look like
`/spatialblend.scan.v1.ReconstructionService/Segment`. `distributed.proto` is the
repository's top-level `proto/distributed.proto`, shared with the server; edit it
there (field numbers are pinned by `server/tests/test_wire_contract.py`).

### TLS

`spatialblend_inference_session_connect_tls` uses `grpc::SslCredentials`; the
server certificate is verified against the host in `host:port`. Roots: the PEM
file in `SPATIALBLEND_TLS_ROOT_CERTS_FILE` if set, else gRPC default discovery
(`GRPC_DEFAULT_SSL_ROOTS_FILE_PATH`, then the OS trust store: Android
`/system/etc/security/cacerts`, macOS `/etc/ssl/cert.pem`). The pinned gRPC
does not compile a roots.pem into the library. Test:
`bazel test //session/inference/sam3d:sam3d_tls_test`.

## Code Style

- **C++17** with Google C++ style guide
- **Files**: `snake_case.{cc,h}`, tests: `snake_case_test.cc`
- **Classes**: `PascalCase`
- **Functions/Variables**: `snake_case`
- **Constants**: `kPascalCase`
- **Private members**: trailing underscore (`member_name_`)
- **Namespaces**: `spatialblend` for main code
