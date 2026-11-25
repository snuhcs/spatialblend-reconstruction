// Copyright (c) 2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SESSION_C_SPATIALBLEND_SESSION_H_
#define SESSION_C_SPATIALBLEND_SESSION_H_

#include <cstdint>

#include "spatialblend_session_type.h"
#include "spatialblend_session_utility.h"

#if defined(_WIN32)
#if defined(SPATIALBLEND_BUILD_DLL)
#define SPATIALBLEND_EXPORT __declspec(dllexport)
#else
#define SPATIALBLEND_EXPORT __declspec(dllimport)
#endif  // SPATIALBLEND_BUILD_DLL
#else
#define SPATIALBLEND_EXPORT __attribute__((visibility("default")))
#endif  // _WIN32

#ifdef __cplusplus
extern "C" {
#endif

/**
 * ============================================================================
 * INFERENCE (External gRPC Service)
 * ============================================================================
 *
 * These APIs connect to an external gRPC server that implements
 * `OneShotReconstructionService` defined in `session/inference/sam3d/proto/sam3d.proto`
 * (kept wire-compatible with the existing Python server).
 */

/**
 * @brief Callback invoked when an inference operation completes.
 *
 * @param session_handle The inference session that completed the operation
 * @param event_handle The event handle returned by the async call
 * @param success true if the operation succeeded
 * @param message Empty if success==true, otherwise an error string
 */
typedef void (*sb_on_inference_complete_t)(sb_inference_session_t session_handle,
                                           sb_handle_t event_handle,
                                           bool success, const char* message);

SPATIALBLEND_EXPORT sb_inference_session_t spatialblend_inference_session_create(
    sb_inference_type_t type);
SPATIALBLEND_EXPORT sb_handle_t spatialblend_inference_session_destroy(
    sb_inference_session_t session_handle);

/**
 * Connect to inference server at "host:port" (e.g. "127.0.0.1:50051").
 */
SPATIALBLEND_EXPORT sb_handle_t spatialblend_inference_session_connect(
    sb_inference_session_t session_handle, const char* addr);

/**
 * Connect to inference server at "host:port" over TLS (e.g.
 * "scan-host.example:443"). Same signature and return contract as
 * spatialblend_inference_session_connect(); only the transport differs.
 *
 * The server certificate is verified against the host part of `addr` (no
 * target-name override, no plaintext fallback). Trust roots:
 *  1. If the environment variable SPATIALBLEND_TLS_ROOT_CERTS_FILE names a PEM
 *     file, exactly those roots are used (an unreadable or empty file makes
 *     the connect fail). This is the portable way to supply roots on platforms
 *     whose trust store gRPC cannot read, and to trust a private CA.
 *  2. Otherwise gRPC's default discovery: $GRPC_DEFAULT_SSL_ROOTS_FILE_PATH,
 *     then the OS trust store (Linux/Android: CA bundle files, then
 *     directories such as /system/etc/security/cacerts; macOS:
 *     /etc/ssl/cert.pem). The bundled gRPC does not compile roots.pem in.
 *
 * Like the plaintext variant, the returned event resolves immediately: it
 * reports argument/session errors, not the outcome of the TLS handshake,
 * which happens lazily on the first RPC (a certificate failure surfaces as
 * that request's error).
 */
SPATIALBLEND_EXPORT sb_handle_t spatialblend_inference_session_connect_tls(
    sb_inference_session_t session_handle, const char* addr);

/**
 * Send async ProcessImage RPC (completion tracked by returned event handle).
 *
 * - `client_id`: caller-supplied identifier forwarded to the server
 * - `image_data`: any bytes (server only logs the byte length)
 * - `points`: 3D points (x,y,z) forwarded to server
 * - `output_path`: if non-null, used as a base prefix (no extension).
 *   The GLB model is saved as `<output_path>_objmodel.glb` and the JSON
 *   obj info is saved as `<output_path>_objinfo.json`.
 *
 * When the RPC completes, the callback registered via
 * `spatialblend_register_on_inference_complete()` is automatically invoked
 * from a background thread.
 */
SPATIALBLEND_EXPORT sb_handle_t spatialblend_inference_sam3d(
    sb_inference_session_t session_handle, const char* client_id,
    const void* image_data, uint64_t image_data_size,
    const sb_point_3d_t* points, uint64_t num_points,
    const char* output_path);

/**
 * Send a generic async inference request.
 *
 * @param session_handle  Active inference session
 * @param client_id       Caller-supplied identifier forwarded to the server
 * @param input           Input parameters (must not be null)
 * @return Event handle for completion tracking
 */
SPATIALBLEND_EXPORT sb_handle_t spatialblend_inference_request(
    sb_inference_session_t session_handle, const char* client_id,
    const sb_inference_input_t* input);

/**
 * Register a callback to be invoked when inference operations complete.
 *
 * @param session_handle The inference session to register the callback for
 * @param callback The callback function to invoke on completion
 * @return true if registration succeeded, false otherwise
 */
SPATIALBLEND_EXPORT bool spatialblend_register_on_inference_complete(
    sb_inference_session_t session_handle, sb_on_inference_complete_t callback);

/**
 * Send async Segment RPC (SAM3 segmentation only).
 * Session must be created with SB_INFERENCE_TYPE_SAM3_SEG.
 *
 * - `output_mask_path`: if non-null, raw mask bytes are saved to this path.
 *   Mask layout: mask_height * mask_width bytes, uint8, 0=background 1=foreground.
 *
 * The registered callback fires on completion with success/failure status.
 */
SPATIALBLEND_EXPORT sb_handle_t spatialblend_inference_segment(
    sb_inference_session_t session_handle, const char* client_id,
    const void* image_data, uint64_t image_data_size,
    const sb_point_3d_t* points, uint64_t num_points,
    const char* output_mask_path);

/**
 * Send async Reconstruct RPC (SAM3D-Object reconstruction only).
 * Session must be created with SB_INFERENCE_TYPE_SAM3D_OBJ.
 *
 * - `mask_data`: raw mask bytes from a prior Segment call.
 * - `mask_width`, `mask_height`: mask dimensions.
 * - `output_path`: if non-null, used as a base prefix (no extension).
 *   GLB -> `<output_path>_objmodel.glb`, JSON -> `<output_path>_objinfo.json`.
 */
SPATIALBLEND_EXPORT sb_handle_t spatialblend_inference_reconstruct(
    sb_inference_session_t session_handle, const char* client_id,
    const void* image_data, uint64_t image_data_size,
    const void* mask_data, uint64_t mask_data_size,
    int32_t mask_width, int32_t mask_height,
    const char* output_path);

typedef void (*sb_log_callback_t)(sb_log_severity_t severity,
                                  const char* message);

/**
 * @brief Register a callback to receive log messages
 */
SPATIALBLEND_EXPORT sb_handle_t spatialblend_register_log_listener(
    sb_log_severity_t verbosity, sb_log_callback_t callback);

/**
 * @brief Unregister a log listener callback
 */
SPATIALBLEND_EXPORT bool spatialblend_unregister_log_listener(
    sb_handle_t callback_id);

/**
 * @brief Clean up all inference sessions and log listeners
 */
SPATIALBLEND_EXPORT void spatialblend_cleanup_all();

#ifdef __cplusplus
}
#endif

#endif  // SESSION_C_SPATIALBLEND_SESSION_H_
