// Copyright (c) 2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SESSION_INFERENCE_INFERENCE_CLIENT_H_
#define SESSION_INFERENCE_INFERENCE_CLIENT_H_

#include <atomic>
#include <cstddef>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include <grpcpp/security/credentials.h>
#include "session/network/grpc/grpc_client_base.h"

namespace spatialblend::inference {

struct InferenceResult {
  bool success = false;
  std::string message;
  std::string output_data;  // GLB bytes (Reconstruct) or raw mask bytes (Segment)
  std::string json_data;    // JSON metadata (Reconstruct only)
  // Segment only: mask dimensions for reshaping output_data
  // output_data layout: mask_height * mask_width bytes, uint8, 0=bg 1=fg
  int32_t mask_width = 0;
  int32_t mask_height = 0;
};

// Single base class for all inference clients. Merges gRPC async connection
// management with the pure virtual inference interface into one class.
//
// Inherits IGrpcClient for enable_shared_from_this and pending-call tracking,
// which are needed by GrpcClientCallBase-derived CQ tags.
//
// Concrete methods:
//   Connect(), Disconnect(), IsConnected(), HandleRpcs()
//   AddPendingCall(), RemovePendingCall()
//
// Subclasses must implement:
//   ConnectToAddress() -- address parsing + call to Connect()
//   Process()          -- model-specific RPC dispatch
//   OnConnect()        -- stub creation (optional, default is no-op)
//
// Transport security: plaintext by default (local tests, trusted LAN). Call
// SetTransportSecurity(TransportSecurity::kTls) before ConnectToAddress() /
// Connect() to use TLS with the server certificate verified against the host
// name in the address. See CreateChannelCredentials() for how roots are found.
class InferenceClientBase : public internal::IGrpcClient {
 public:
  enum class TransportSecurity { kInsecure, kTls };

  // Optional PEM root-certificate file used instead of gRPC's default root
  // discovery when TLS is selected (see CreateChannelCredentials()).
  static constexpr const char* kTlsRootCertsFileEnvVar =
      "SPATIALBLEND_TLS_ROOT_CERTS_FILE";

  InferenceClientBase() = default;
  ~InferenceClientBase() override;

  // --- Connection management (concrete) ---
  absl::Status Connect(const std::string& address, int port);
  absl::Status Connect(const std::string& address, int port,
                       const grpc::ChannelArguments& channel_args);
  absl::Status Disconnect();
  bool IsConnected() const;

  // Selects plaintext (default) or TLS for the next Connect(). Has no effect
  // on an already-connected client.
  void SetTransportSecurity(TransportSecurity security) {
    transport_security_ = security;
  }
  TransportSecurity transport_security() const { return transport_security_; }

  // Builds the channel credentials for `security`.
  //
  //  kInsecure: grpc::InsecureChannelCredentials().
  //  kTls:      grpc::SslCredentials(). Trust roots, in order:
  //             1. the PEM file named by $SPATIALBLEND_TLS_ROOT_CERTS_FILE, if
  //                set (an unreadable or empty file is an error, never a
  //                silent fallback to other roots);
  //             2. gRPC's default discovery: $GRPC_DEFAULT_SSL_ROOTS_FILE_PATH,
  //                then the OS trust store (Linux/Android: well-known CA
  //                bundle files, then directories such as
  //                /system/etc/security/cacerts; macOS: /etc/ssl/cert.pem).
  //             The gRPC build used here does not compile a roots.pem into the
  //             library. The server certificate is always verified against the
  //             host name used to connect; there is no target-name override.
  static absl::StatusOr<std::shared_ptr<grpc::ChannelCredentials>>
  CreateChannelCredentials(TransportSecurity security);

  // --- Inference interface (pure virtual) ---

  virtual absl::Status ConnectToAddress(
      const std::string& address_with_port) = 0;

  // @param client_id  Unique request identifier
  // @param input_data Primary input bytes (e.g. image data)
  // @param params     Optional float parameters (model-specific;
  //                   SAM3D: flattened xyz triplets [x1,y1,z1,...])
  // @param params_size Number of float values in params
  virtual std::future<absl::StatusOr<InferenceResult>> Process(
      const std::string& client_id,
      const std::string& input_data,
      const float* params, size_t params_size) = 0;

 protected:
  // Called after channel and CQ are created but before HandleRpcs starts.
  virtual absl::Status OnConnect() { return absl::OkStatus(); }
  void HandleRpcs();

  // IGrpcClient interface for CQ call lifetime management.
  void AddPendingCall(GrpcClientCallBase* call) override;
  void RemovePendingCall(GrpcClientCallBase* call) override;

  // gRPC connection state
  TransportSecurity transport_security_ = TransportSecurity::kInsecure;
  std::shared_ptr<grpc::Channel> channel_;
  std::unique_ptr<grpc::CompletionQueue> cq_;
  std::thread async_thread_;
  std::atomic<bool> shutdown_{false};
  std::atomic<bool> connected_{false};
  std::mutex pending_calls_mutex_;
  std::unordered_set<GrpcClientCallBase*> pending_calls_;
};

}  // namespace spatialblend::inference

#endif  // SESSION_INFERENCE_INFERENCE_CLIENT_H_
