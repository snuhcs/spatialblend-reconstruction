// Copyright (c) 2026 Seoul National University
// SPDX-License-Identifier: MIT

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "session/c/spatialblend_session.h"
#include "session/c/spatialblend_session_utility.h"

// Global failure tracking
static bool test_failed = false;

void test_inference_session_operations() {
  printf("\n=== Testing Inference Session Operations ===\n");

  // Test 1: Handle creation and uniqueness
  sb_inference_session_t s1 = spatialblend_inference_session_create(SB_INFERENCE_TYPE_SAM3D);
  sb_inference_session_t s2 = spatialblend_inference_session_create(SB_INFERENCE_TYPE_SAM3D);
  if (s1 == 0 || s2 == 0) {
    printf("FAILED: inference session creation returned 0\n");
    test_failed = true;
    return;
  }
  if (s1 == s2) {
    printf("FAILED: inference session handles should be unique\n");
    test_failed = true;
    return;
  }
  printf("  Inference session handles are unique (s1=%lld, s2=%lld)\n",
         (long long)s1, (long long)s2);

  // Test 1b: Invalid type returns 0
  sb_inference_session_t s_bad_type =
      spatialblend_inference_session_create(static_cast<sb_inference_type_t>(999));
  if (s_bad_type != 0) {
    printf("FAILED: invalid inference type should return 0\n");
    test_failed = true;
    return;
  }
  printf("  Invalid inference type correctly returns 0\n");

  // Test 2: Callback registration with valid session
  bool reg_ok = spatialblend_register_on_inference_complete(
      s1, [](sb_inference_session_t, sb_handle_t, bool, const char*) {});
  if (!reg_ok) {
    printf("FAILED: callback registration should succeed for valid session\n");
    test_failed = true;
    return;
  }
  printf("  Callback registration succeeded for valid session\n");

  // Test 3: Callback registration with invalid session (0)
  sb_inference_session_t invalid_session = 0;
  bool reg_bad = spatialblend_register_on_inference_complete(
      invalid_session, [](sb_inference_session_t, sb_handle_t, bool, const char*) {});
  if (reg_bad) {
    printf("FAILED: callback registration should fail for invalid session\n");
    test_failed = true;
    return;
  }
  printf("  Callback registration correctly rejected for invalid session\n");

  // Test 4: Connect with null address
  sb_handle_t h_null_addr = spatialblend_inference_session_connect(s1, nullptr);
  if (h_null_addr != 0) {
    bool ok = spatialblend_try_wait_for_event(h_null_addr, 1000);
    printf("  Connect with null address returned event (resolved=%s)\n",
           ok ? "true" : "false");
  } else {
    printf("  Connect with null address returned handle 0\n");
  }

  // Test 5: Connect with invalid session
  sb_handle_t h_bad_session = spatialblend_inference_session_connect(invalid_session,
                                                                   "127.0.0.1:50051");
  if (h_bad_session != 0) {
    spatialblend_try_wait_for_event(h_bad_session, 1000);
  }
  printf("  Connect with invalid session handled gracefully\n");

  // Test 6: ProcessImage with invalid session (legacy wrapper)
  uint8_t dummy_data[] = {0x00, 0x01};
  sb_handle_t h_bad_proc = spatialblend_inference_sam3d(
      invalid_session, "test", dummy_data, sizeof(dummy_data), nullptr, 0, nullptr);
  if (h_bad_proc != 0) {
    spatialblend_try_wait_for_event(h_bad_proc, 1000);
  }
  printf("  ProcessImage with invalid session handled gracefully\n");

  // Test 7: Generic process with null input
  sb_inference_session_t s3 = spatialblend_inference_session_create(SB_INFERENCE_TYPE_SAM3D);
  sb_handle_t h_null_input = spatialblend_inference_request(s3, "test", nullptr);
  if (h_null_input != 0) {
    spatialblend_try_wait_for_event(h_null_input, 1000);
  }
  printf("  Generic process with null input handled gracefully\n");

  // Test 8: Generic process on unconnected session
  sb_inference_input_t input = {};
  input.data = dummy_data;
  input.data_size = sizeof(dummy_data);
  sb_handle_t h_not_conn = spatialblend_inference_request(s3, "test", &input);
  if (h_not_conn != 0) {
    spatialblend_try_wait_for_event(h_not_conn, 1000);
  }
  printf("  Generic process on unconnected session handled gracefully\n");

  // Test 9: ProcessImage on unconnected session (legacy wrapper)
  sb_handle_t h_not_conn2 = spatialblend_inference_sam3d(
      s3, "test", dummy_data, sizeof(dummy_data), nullptr, 0, nullptr);
  if (h_not_conn2 != 0) {
    spatialblend_try_wait_for_event(h_not_conn2, 1000);
  }
  printf("  ProcessImage on unconnected session handled gracefully\n");

  // Test 10: ProcessImage with NULL image data
  sb_handle_t h_null_data = spatialblend_inference_sam3d(
      s3, "test", nullptr, 0, nullptr, 0, nullptr);
  if (h_null_data != 0) {
    spatialblend_try_wait_for_event(h_null_data, 1000);
  }
  printf("  ProcessImage with NULL data handled gracefully\n");

  // Test 11: Destroy valid session
  sb_handle_t h_destroy1 = spatialblend_inference_session_destroy(s1);
  if (h_destroy1 != 0) {
    bool ok = spatialblend_try_wait_for_event(h_destroy1, 1000);
    if (!ok) {
      printf("FAILED: destroy of valid session should succeed\n");
      test_failed = true;
      return;
    }
  }
  printf("  Inference session s1 destroyed successfully\n");

  // Test 12: Destroy with 0 (should return error event, not crash)
  sb_handle_t h_null_destroy = spatialblend_inference_session_destroy(0);
  if (h_null_destroy != 0) {
    spatialblend_try_wait_for_event(h_null_destroy, 1000);
  }
  printf("  Destroy with 0 handled gracefully (no crash)\n");

  // Clean up remaining sessions
  spatialblend_inference_session_destroy(s2);
  spatialblend_inference_session_destroy(s3);

  printf("  Inference session operations test completed\n");
}

void test_inference_sam3d_with_output_path() {
  printf("\n=== Testing spatialblend_inference_sam3d with output_path ===\n");

  sb_inference_session_t s =
      spatialblend_inference_session_create(SB_INFERENCE_TYPE_SAM3D);
  if (s == 0) {
    printf("FAILED: session creation returned 0\n");
    test_failed = true;
    return;
  }

  // Test: spatialblend_inference_sam3d with non-null output_path on unconnected
  // session. output_path is now a base prefix: GLB -> <path>_objmodel.glb,
  // JSON -> <path>_objinfo.json.
  uint8_t dummy_data[] = {0x89, 0x50, 0x4E, 0x47};  // PNG header bytes
  sb_handle_t h = spatialblend_inference_sam3d(
      s, "test_user", dummy_data, sizeof(dummy_data),
      nullptr, 0, "/tmp/c_api_test_output");
  if (h != 0) {
    spatialblend_try_wait_for_event(h, 2000);
  }
  printf("  spatialblend_inference_sam3d with output_path handled gracefully\n");

  spatialblend_inference_session_destroy(s);
  printf("  output_path test completed\n");
}

void test_inference_sam3d_with_points() {
  printf("\n=== Testing spatialblend_inference_sam3d with sb_point_3d_t ===\n");

  sb_inference_session_t s =
      spatialblend_inference_session_create(SB_INFERENCE_TYPE_SAM3D);
  if (s == 0) {
    printf("FAILED: session creation returned 0\n");
    test_failed = true;
    return;
  }

  // Test: sb_point_3d_t (double) -> flat float conversion path
  uint8_t dummy_data[] = {0x89, 0x50, 0x4E, 0x47};
  sb_point_3d_t points[] = {
      {100.5, 200.3, 0.0},
      {150.7, 250.1, 0.0},
      {300.0, 400.0, 0.0},
  };

  sb_handle_t h = spatialblend_inference_sam3d(
      s, "test_user", dummy_data, sizeof(dummy_data),
      points, 3, nullptr);
  if (h != 0) {
    spatialblend_try_wait_for_event(h, 2000);
  }
  printf("  spatialblend_inference_sam3d with points handled gracefully\n");

  // Test: points + output_path together
  sb_handle_t h2 = spatialblend_inference_sam3d(
      s, "test_user", dummy_data, sizeof(dummy_data),
      points, 3, "/tmp/c_api_test_points_output");
  if (h2 != 0) {
    spatialblend_try_wait_for_event(h2, 2000);
  }
  printf("  spatialblend_inference_sam3d with points + output_path handled gracefully\n");

  spatialblend_inference_session_destroy(s);
  printf("  points test completed\n");
}

void test_inference_request_with_output_path() {
  printf("\n=== Testing spatialblend_inference_request with output_path ===\n");

  sb_inference_session_t s =
      spatialblend_inference_session_create(SB_INFERENCE_TYPE_SAM3D);
  if (s == 0) {
    printf("FAILED: session creation returned 0\n");
    test_failed = true;
    return;
  }

  // Test: sb_inference_input_t with output_path set
  uint8_t dummy_data[] = {0x89, 0x50, 0x4E, 0x47};
  float params[] = {100.5f, 200.3f, 0.0f, 150.7f, 250.1f, 0.0f};

  sb_inference_input_t input = {};
  input.data = dummy_data;
  input.data_size = sizeof(dummy_data);
  input.params = params;
  input.num_params = 6;
  input.output_path = "/tmp/c_api_test_generic_output";

  sb_handle_t h = spatialblend_inference_request(s, "test_user", &input);
  if (h != 0) {
    spatialblend_try_wait_for_event(h, 2000);
  }
  printf("  spatialblend_inference_request with output_path handled gracefully\n");

  // Test: output_path with empty string (should behave like nullptr)
  sb_inference_input_t input_empty = {};
  input_empty.data = dummy_data;
  input_empty.data_size = sizeof(dummy_data);
  input_empty.output_path = "";

  sb_handle_t h2 = spatialblend_inference_request(s, "test_user", &input_empty);
  if (h2 != 0) {
    spatialblend_try_wait_for_event(h2, 2000);
  }
  printf("  spatialblend_inference_request with empty output_path handled gracefully\n");

  spatialblend_inference_session_destroy(s);
  printf("  generic request output_path test completed\n");
}

static void set_roots_env(const char* value) {
#ifdef _WIN32
  _putenv_s("SPATIALBLEND_TLS_ROOT_CERTS_FILE", value ? value : "");
#else
  if (value) {
    setenv("SPATIALBLEND_TLS_ROOT_CERTS_FILE", value, 1);
  } else {
    unsetenv("SPATIALBLEND_TLS_ROOT_CERTS_FILE");
  }
#endif
}

static bool event_failed_with(sb_handle_t h, const char* needle) {
  // Note: event handles start at 0, so 0 is a valid handle.
  if (spatialblend_try_wait_for_event(h, 1000)) return false;
  const char* msg = spatialblend_get_error_message(h);
  return msg != nullptr && strstr(msg, needle) != nullptr;
}

void test_inference_session_connect_tls() {
  printf("\n=== Testing spatialblend_inference_session_connect_tls ===\n");

  sb_inference_session_t s = spatialblend_inference_session_create(SB_INFERENCE_TYPE_SAM3D);
  if (s == 0) {
    printf("FAILED: session creation\n");
    test_failed = true;
    return;
  }

  // Same error contract as spatialblend_inference_session_connect.
  if (!event_failed_with(spatialblend_inference_session_connect_tls(s, nullptr),
                         "addr must not be null")) {
    printf("FAILED: connect_tls(null addr) should report InvalidArgument\n");
    test_failed = true;
  }
  if (!event_failed_with(spatialblend_inference_session_connect_tls(0, "localhost:443"),
                         "Invalid inference session")) {
    printf("FAILED: connect_tls(invalid session) should report InvalidArgument\n");
    test_failed = true;
  }
  if (!event_failed_with(spatialblend_inference_session_connect_tls(s, "no-port"),
                         "")) {
    printf("FAILED: connect_tls(address without port) should fail\n");
    test_failed = true;
  }
  if (!event_failed_with(spatialblend_inference_session_connect_tls(s, "a..b..c:443"),
                         "Invalid or unreachable host address")) {
    printf("FAILED: connect_tls(malformed host name) should fail\n");
    test_failed = true;
  }

  // A host that looks like a gRPC URI scheme ("unix:443") is a host name, not
  // a scheme: the connect is accepted (the channel is lazy) and uses dns:///.
  {
    sb_handle_t hu = spatialblend_inference_session_connect_tls(s, "unix:443");
    if (!spatialblend_try_wait_for_event(hu, 1000)) {
      printf("FAILED: connect_tls(\"unix:443\") should be accepted\n");
      test_failed = true;
    }
    spatialblend_inference_session_destroy(s);
    s = spatialblend_inference_session_create(SB_INFERENCE_TYPE_SAM3D);
  }

  // An unreadable root-certificate override fails closed, before any channel
  // is created (and without silently falling back to other roots).
  set_roots_env("/nonexistent/spatialblend-test-roots.pem");
  if (!event_failed_with(
          spatialblend_inference_session_connect_tls(s, "scan-host.example:443"),
          "SPATIALBLEND_TLS_ROOT_CERTS_FILE")) {
    printf("FAILED: unreadable root override should fail the connect\n");
    test_failed = true;
  }
  set_roots_env(nullptr);

  // The plaintext path is unaffected by the earlier TLS attempts.
  sb_handle_t h = spatialblend_inference_session_connect(s, "127.0.0.1:50051");
  if (!spatialblend_try_wait_for_event(h, 1000)) {
    printf("FAILED: plaintext connect to loopback should still succeed\n");
    test_failed = true;
  }
  spatialblend_inference_session_destroy(s);
  printf("  connect_tls error contract and plaintext path OK\n");
}

int main() {
  printf("=== SpatialBlend Session C API Test ===\n\n");

  test_inference_session_connect_tls();
  test_inference_session_operations();
  test_inference_sam3d_with_output_path();
  test_inference_sam3d_with_points();
  test_inference_request_with_output_path();

  printf("\n=== All tests completed! ===\n");

  if (test_failed) {
    printf("FAILED: Some tests failed!\n");
    return 1;
  } else {
    printf("PASSED: All tests passed!\n");
    return 0;
  }
}
