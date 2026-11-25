// Copyright (c) 2026 Seoul National University
// SPDX-License-Identifier: MIT

#include <atomic>
#include <cstring>
#include <fstream>
#include <future>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "session/c/spatialblend_session.h"
#include "session/inference/inference_client.h"
#include "session/inference/distributed/sam3_seg_client.h"
#include "session/inference/distributed/sam3d_obj_client.h"
#include "session/inference/sam3d/sam3d_client.h"
#include "session/util/logger.h"
#include "session/util/status_ring.h"
#include "session/util/util.h"

using namespace spatialblend;

namespace {

// --- Event ring ---
spatialblend::StatusRing g_status_ring(1000);

sb_handle_t StoreEvent(std::future<absl::Status> future) {
  return g_status_ring.Push(std::move(future));
}

// --- Log listener tracking ---
std::vector<sb_handle_t> g_log_listener_handles;

// --- Log severity conversions ---
sb_log_severity_t ConvertLogSeverity(LogSeverity severity) {
  switch (severity) {
    case LogSeverity::kInfo:
      return SB_LOG_SEVERITY_INFO;
    case LogSeverity::kWarning:
      return SB_LOG_SEVERITY_WARNING;
    case LogSeverity::kError:
      return SB_LOG_SEVERITY_ERROR;
    default:
      return SB_LOG_SEVERITY_INFO;
  }
}

LogSeverity ConvertLogSeverity(sb_log_severity_t severity) {
  switch (severity) {
    case SB_LOG_SEVERITY_INFO:
      return LogSeverity::kInfo;
    case SB_LOG_SEVERITY_WARNING:
      return LogSeverity::kWarning;
    case SB_LOG_SEVERITY_ERROR:
      return LogSeverity::kError;
    default:
      return LogSeverity::kInfo;
  }
}

// --- Inference session registry (handle-based) ---
struct InferenceSession {
  std::shared_ptr<spatialblend::inference::InferenceClientBase> client;
  sb_inference_type_t type;
  std::string connected_addr;
  sb_on_inference_complete_t on_complete_callback = nullptr;
  std::mutex callback_mutex;
};

std::unordered_map<int64_t, std::shared_ptr<InferenceSession>>
    g_inference_sessions;
std::mutex g_inference_sessions_mutex;
std::atomic<int64_t> g_next_inference_handle{1};

std::shared_ptr<InferenceSession> GetInferenceSession(
    sb_inference_session_t handle) {
  std::lock_guard<std::mutex> lock(g_inference_sessions_mutex);
  auto it = g_inference_sessions.find(handle);
  if (it == g_inference_sessions.end()) return nullptr;
  return it->second;
}

}  // namespace

extern "C" {

sb_inference_session_t spatialblend_inference_session_create(
    sb_inference_type_t type) {
  auto session = std::make_shared<InferenceSession>();
  session->type = type;

  switch (type) {
    case SB_INFERENCE_TYPE_SAM3D:
      session->client = std::make_shared<spatialblend::sam3d::Sam3dClient>();
      break;
    case SB_INFERENCE_TYPE_SAM3_SEG:
      session->client =
          std::make_shared<spatialblend::distributed::Sam3SegClient>();
      break;
    case SB_INFERENCE_TYPE_SAM3D_OBJ:
      session->client =
          std::make_shared<spatialblend::distributed::Sam3dObjClient>();
      break;
    default:
      return 0;
  }

  int64_t handle = g_next_inference_handle.fetch_add(1);
  {
    std::lock_guard<std::mutex> lock(g_inference_sessions_mutex);
    g_inference_sessions[handle] = std::move(session);
  }
  return handle;
}

sb_handle_t spatialblend_inference_session_destroy(
    sb_inference_session_t session_handle) {
  std::shared_ptr<InferenceSession> session;
  {
    std::lock_guard<std::mutex> lock(g_inference_sessions_mutex);
    auto it = g_inference_sessions.find(session_handle);
    if (it == g_inference_sessions.end()) {
      return StoreEvent(util::MakeReadyFuture(
          absl::InvalidArgumentError("Invalid inference session")));
    }
    session = std::move(it->second);
    g_inference_sessions.erase(it);
  }

  if (session->client) {
    (void)session->client->Disconnect();
  }
  return StoreEvent(util::MakeReadyFuture(absl::OkStatus()));
}

namespace {

sb_handle_t ConnectInferenceSession(
    sb_inference_session_t session_handle, const char* addr,
    spatialblend::inference::InferenceClientBase::TransportSecurity security) {
  if (!addr) {
    return StoreEvent(util::MakeReadyFuture(
        absl::InvalidArgumentError("addr must not be null")));
  }
  auto session = GetInferenceSession(session_handle);
  if (!session || !session->client) {
    return StoreEvent(util::MakeReadyFuture(
        absl::InvalidArgumentError("Invalid inference session")));
  }

  session->connected_addr = addr;
  session->client->SetTransportSecurity(security);
  absl::Status st = session->client->ConnectToAddress(session->connected_addr);
  // Do not let a TLS attempt change what a later plaintext connect does.
  session->client->SetTransportSecurity(
      spatialblend::inference::InferenceClientBase::TransportSecurity::kInsecure);
  return StoreEvent(util::MakeReadyFuture(st));
}

}  // namespace

sb_handle_t spatialblend_inference_session_connect(
    sb_inference_session_t session_handle, const char* addr) {
  return ConnectInferenceSession(
      session_handle, addr,
      spatialblend::inference::InferenceClientBase::TransportSecurity::kInsecure);
}

sb_handle_t spatialblend_inference_session_connect_tls(
    sb_inference_session_t session_handle, const char* addr) {
  return ConnectInferenceSession(
      session_handle, addr,
      spatialblend::inference::InferenceClientBase::TransportSecurity::kTls);
}

sb_handle_t spatialblend_inference_request(
    sb_inference_session_t session_handle, const char* client_id,
    const sb_inference_input_t* input) {
  if (!input) {
    return StoreEvent(util::MakeReadyFuture(
        absl::InvalidArgumentError("input must not be null")));
  }

  auto session = GetInferenceSession(session_handle);
  if (!session || !session->client) {
    return StoreEvent(util::MakeReadyFuture(
        absl::InvalidArgumentError("Invalid inference session")));
  }

  const std::string cid = client_id ? client_id : "";

  std::string input_data;
  if (input->data && input->data_size > 0) {
    input_data.assign(static_cast<const char*>(input->data),
                      static_cast<size_t>(input->data_size));
  }

  const std::string output_path_copy =
      input->output_path ? std::string(input->output_path) : "";

  std::future<absl::StatusOr<spatialblend::inference::InferenceResult>> rpc_future =
      session->client->Process(cid, input_data,
                               input->params,
                               static_cast<size_t>(input->num_params));

  // Pre-allocate the event handle so we can pass it to the callback.
  std::promise<absl::Status> status_promise;
  std::future<absl::Status> status_future = status_promise.get_future();
  sb_handle_t event_handle = StoreEvent(std::move(status_future));

  std::thread([session, session_handle, event_handle, output_path_copy,
               rpc_future = std::move(rpc_future),
               status_promise = std::move(status_promise)]() mutable {
    absl::StatusOr<spatialblend::inference::InferenceResult> rpc = rpc_future.get();
    absl::Status final_status;
    std::string error_message;

    if (!rpc.ok()) {
      final_status = rpc.status();
      error_message = final_status.ToString();
    } else if (!rpc->success) {
      final_status = absl::InternalError(
          absl::StrFormat("Inference server failure: %s", rpc->message));
      error_message = rpc->message;
    } else {
      // RPC succeeded, handle optional file output.
      // output_path is now a base prefix (no extension).
      // GLB -> <output_path>_objmodel.glb
      // JSON -> <output_path>_objinfo.json
      if (!output_path_copy.empty()) {
        const std::string glb_path = output_path_copy + "_objmodel.glb";
        const std::string json_path = output_path_copy + "_objinfo.json";

        // Write GLB
        {
          std::ofstream out(glb_path, std::ios::binary);
          if (!out) {
            final_status = absl::InternalError(
                absl::StrFormat("Failed to open GLB output: %s", glb_path));
            error_message = final_status.ToString();
          } else {
            out.write(rpc->output_data.data(),
                      static_cast<std::streamsize>(rpc->output_data.size()));
            out.close();
            if (!out) {
              final_status = absl::InternalError(
                  absl::StrFormat("Failed to write GLB output: %s", glb_path));
              error_message = final_status.ToString();
            }
          }
        }

        // Write JSON (only if GLB succeeded)
        if (final_status.ok() && !rpc->json_data.empty()) {
          std::ofstream out(json_path, std::ios::binary);
          if (!out) {
            final_status = absl::InternalError(
                absl::StrFormat("Failed to open JSON output: %s", json_path));
            error_message = final_status.ToString();
          } else {
            out.write(rpc->json_data.data(),
                      static_cast<std::streamsize>(rpc->json_data.size()));
            out.close();
            if (!out) {
              final_status = absl::InternalError(
                  absl::StrFormat("Failed to write JSON output: %s", json_path));
              error_message = final_status.ToString();
            }
          }
        }

        if (final_status.ok()) {
          final_status = absl::OkStatus();
        }
      } else {
        final_status = absl::OkStatus();
      }
    }

    // Invoke the registered callback
    {
      std::lock_guard<std::mutex> lock(session->callback_mutex);
      if (session->on_complete_callback) {
        session->on_complete_callback(session_handle, event_handle,
                                      final_status.ok(),
                                      error_message.c_str());
      }
    }

    // Complete the event handle for legacy polling support
    status_promise.set_value(final_status);
  }).detach();

  return event_handle;
}

sb_handle_t spatialblend_inference_sam3d(
    sb_inference_session_t session_handle, const char* client_id,
    const void* image_data, uint64_t image_data_size,
    const sb_point_3d_t* points, uint64_t num_points,
    const char* output_path) {
  // Convert sb_point_3d_t (double) to flat float array for generic API
  std::vector<float> flat_params;
  if (points && num_points > 0) {
    flat_params.reserve(num_points * 3);
    for (uint64_t i = 0; i < num_points; ++i) {
      flat_params.push_back(static_cast<float>(points[i].x));
      flat_params.push_back(static_cast<float>(points[i].y));
      flat_params.push_back(static_cast<float>(points[i].z));
    }
  }

  sb_inference_input_t input = {};
  input.data = image_data;
  input.data_size = image_data_size;
  input.params = flat_params.empty() ? nullptr : flat_params.data();
  input.num_params = static_cast<uint64_t>(flat_params.size());
  input.output_path = output_path;

  return spatialblend_inference_request(session_handle, client_id, &input);
}

sb_handle_t spatialblend_inference_segment(
    sb_inference_session_t session_handle, const char* client_id,
    const void* image_data, uint64_t image_data_size,
    const sb_point_3d_t* points, uint64_t num_points,
    const char* output_mask_path) {
  auto session = GetInferenceSession(session_handle);
  if (!session || !session->client) {
    return StoreEvent(util::MakeReadyFuture(
        absl::InvalidArgumentError("Invalid inference session")));
  }

  // Convert sb_point_3d_t (double) to flat float [x, y, 0, ...]
  std::vector<float> flat_params;
  if (points && num_points > 0) {
    flat_params.reserve(num_points * 3);
    for (uint64_t i = 0; i < num_points; ++i) {
      flat_params.push_back(static_cast<float>(points[i].x));
      flat_params.push_back(static_cast<float>(points[i].y));
      flat_params.push_back(0.0f);
    }
  }

  std::string image_str;
  if (image_data && image_data_size > 0) {
    image_str.assign(static_cast<const char*>(image_data),
                     static_cast<size_t>(image_data_size));
  }

  const std::string cid = client_id ? client_id : "";
  const std::string mask_path_copy =
      output_mask_path ? std::string(output_mask_path) : "";

  std::future<absl::StatusOr<spatialblend::inference::InferenceResult>> rpc_future =
      session->client->Process(cid, image_str,
                               flat_params.empty() ? nullptr : flat_params.data(),
                               flat_params.size());

  std::promise<absl::Status> status_promise;
  std::future<absl::Status> status_future = status_promise.get_future();
  sb_handle_t event_handle = StoreEvent(std::move(status_future));

  std::thread([session, session_handle, event_handle, mask_path_copy,
               rpc_future = std::move(rpc_future),
               status_promise = std::move(status_promise)]() mutable {
    absl::StatusOr<spatialblend::inference::InferenceResult> rpc = rpc_future.get();
    absl::Status final_status;
    std::string error_message;

    if (!rpc.ok()) {
      final_status = rpc.status();
      error_message = final_status.ToString();
    } else if (!rpc->success) {
      final_status = absl::InternalError(
          absl::StrFormat("Segment server failure: %s", rpc->message));
      error_message = rpc->message;
    } else {
      if (!mask_path_copy.empty() && !rpc->output_data.empty()) {
        std::ofstream out(mask_path_copy, std::ios::binary);
        if (!out) {
          final_status = absl::InternalError(
              absl::StrFormat("Failed to open mask output: %s", mask_path_copy));
          error_message = final_status.ToString();
        } else {
          out.write(rpc->output_data.data(),
                    static_cast<std::streamsize>(rpc->output_data.size()));
          out.close();
          if (!out) {
            final_status = absl::InternalError(absl::StrFormat(
                "Failed to write mask output: %s", mask_path_copy));
            error_message = final_status.ToString();
          }
        }
      }
      if (final_status.ok()) final_status = absl::OkStatus();
    }

    {
      std::lock_guard<std::mutex> lock(session->callback_mutex);
      if (session->on_complete_callback) {
        session->on_complete_callback(session_handle, event_handle,
                                      final_status.ok(),
                                      error_message.c_str());
      }
    }
    status_promise.set_value(final_status);
  }).detach();

  return event_handle;
}

sb_handle_t spatialblend_inference_reconstruct(
    sb_inference_session_t session_handle, const char* client_id,
    const void* image_data, uint64_t image_data_size,
    const void* mask_data, uint64_t mask_data_size,
    int32_t mask_width, int32_t mask_height,
    const char* output_path) {
  auto session = GetInferenceSession(session_handle);
  if (!session || !session->client) {
    return StoreEvent(util::MakeReadyFuture(
        absl::InvalidArgumentError("Invalid inference session")));
  }

  auto obj_client = std::dynamic_pointer_cast<
      spatialblend::distributed::Sam3dObjClient>(session->client);
  if (!obj_client) {
    return StoreEvent(util::MakeReadyFuture(absl::InvalidArgumentError(
        "Session must be created with SB_INFERENCE_TYPE_SAM3D_OBJ")));
  }

  std::string image_str;
  if (image_data && image_data_size > 0) {
    image_str.assign(static_cast<const char*>(image_data),
                     static_cast<size_t>(image_data_size));
  }
  std::string mask_str;
  if (mask_data && mask_data_size > 0) {
    mask_str.assign(static_cast<const char*>(mask_data),
                    static_cast<size_t>(mask_data_size));
  }

  const std::string cid = client_id ? client_id : "";
  const std::string output_path_copy =
      output_path ? std::string(output_path) : "";

  std::future<absl::StatusOr<spatialblend::inference::InferenceResult>> rpc_future =
      obj_client->Reconstruct(cid, image_str, mask_str, mask_width, mask_height);

  std::promise<absl::Status> status_promise;
  std::future<absl::Status> status_future = status_promise.get_future();
  sb_handle_t event_handle = StoreEvent(std::move(status_future));

  std::thread([session, session_handle, event_handle, output_path_copy,
               rpc_future = std::move(rpc_future),
               status_promise = std::move(status_promise)]() mutable {
    absl::StatusOr<spatialblend::inference::InferenceResult> rpc = rpc_future.get();
    absl::Status final_status;
    std::string error_message;

    if (!rpc.ok()) {
      final_status = rpc.status();
      error_message = final_status.ToString();
    } else if (!rpc->success) {
      final_status = absl::InternalError(
          absl::StrFormat("Reconstruct server failure: %s", rpc->message));
      error_message = rpc->message;
    } else {
      if (!output_path_copy.empty()) {
        const std::string glb_path = output_path_copy + "_objmodel.glb";
        const std::string json_path = output_path_copy + "_objinfo.json";

        {
          std::ofstream out(glb_path, std::ios::binary);
          if (!out) {
            final_status = absl::InternalError(
                absl::StrFormat("Failed to open GLB output: %s", glb_path));
            error_message = final_status.ToString();
          } else {
            out.write(rpc->output_data.data(),
                      static_cast<std::streamsize>(rpc->output_data.size()));
            out.close();
            if (!out) {
              final_status = absl::InternalError(
                  absl::StrFormat("Failed to write GLB output: %s", glb_path));
              error_message = final_status.ToString();
            }
          }
        }

        if (final_status.ok() && !rpc->json_data.empty()) {
          std::ofstream out(json_path, std::ios::binary);
          if (!out) {
            final_status = absl::InternalError(
                absl::StrFormat("Failed to open JSON output: %s", json_path));
            error_message = final_status.ToString();
          } else {
            out.write(rpc->json_data.data(),
                      static_cast<std::streamsize>(rpc->json_data.size()));
            out.close();
            if (!out) {
              final_status = absl::InternalError(absl::StrFormat(
                  "Failed to write JSON output: %s", json_path));
              error_message = final_status.ToString();
            }
          }
        }

        if (final_status.ok()) final_status = absl::OkStatus();
      } else {
        final_status = absl::OkStatus();
      }
    }

    {
      std::lock_guard<std::mutex> lock(session->callback_mutex);
      if (session->on_complete_callback) {
        session->on_complete_callback(session_handle, event_handle,
                                      final_status.ok(),
                                      error_message.c_str());
      }
    }
    status_promise.set_value(final_status);
  }).detach();

  return event_handle;
}

bool spatialblend_register_on_inference_complete(
    sb_inference_session_t session_handle,
    sb_on_inference_complete_t callback) {
  auto session = GetInferenceSession(session_handle);
  if (!session) {
    return false;
  }
  std::lock_guard<std::mutex> lock(session->callback_mutex);
  session->on_complete_callback = callback;
  return true;
}

bool spatialblend_try_wait_for_event(sb_handle_t event_handle,
                                  int64_t timeout_ms) {
  auto status = g_status_ring.Get(event_handle, timeout_ms);
  return status.ok();
}

const char* spatialblend_get_error_message(sb_handle_t event_handle) {
  static thread_local std::string error_message;

  auto status = g_status_ring.Get(event_handle, 0);
  if (absl::IsDeadlineExceeded(status)) {
    return "Future not ready";
  }

  if (!status.ok()) {
    error_message = status.ToString();
    return error_message.c_str();
  }

  return "";
}

sb_handle_t spatialblend_register_log_listener(sb_log_severity_t verbosity,
                                            sb_log_callback_t callback) {
  auto severity = ConvertLogSeverity(verbosity);
  auto handle = Logger::Get().SetReporter(
      severity,
      [callback](LogSeverity sev, const char* msg) {
        callback(ConvertLogSeverity(sev), msg);
      });
  g_log_listener_handles.push_back(handle);
  return handle;
}

bool spatialblend_unregister_log_listener(sb_handle_t callback_id) {
  auto status = Logger::Get().RemoveReporter(callback_id);
  if (status.ok()) {
    auto it = std::find(g_log_listener_handles.begin(),
                        g_log_listener_handles.end(), callback_id);
    if (it != g_log_listener_handles.end()) {
      g_log_listener_handles.erase(it);
    }
    return true;
  }
  return false;
}

void spatialblend_cleanup_all() {
  // Unregister all log listeners
  for (sb_handle_t handle : g_log_listener_handles) {
    spatialblend_unregister_log_listener(handle);
  }
  g_log_listener_handles.clear();

  // Destroy all inference sessions
  {
    std::lock_guard<std::mutex> lock(g_inference_sessions_mutex);
    for (auto& [_, session] : g_inference_sessions) {
      if (session && session->client) {
        (void)session->client->Disconnect();
      }
    }
    g_inference_sessions.clear();
  }
}

}  // extern "C"
