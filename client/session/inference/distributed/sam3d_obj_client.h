// Copyright (c) 2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SESSION_INFERENCE_DISTRIBUTED_SAM3D_OBJ_CLIENT_H_
#define SESSION_INFERENCE_DISTRIBUTED_SAM3D_OBJ_CLIENT_H_

#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "session/inference/inference_client.h"
#include "session/inference/distributed/proto/distributed.grpc.pb.h"
#include "session/inference/distributed/proto/distributed.pb.h"

namespace spatialblend::distributed {

struct Sam3dObjClientCall : public GrpcClientCallBase {
  using GrpcClientCallBase::GrpcClientCallBase;
  ~Sam3dObjClientCall() override = default;
};

// Async gRPC client for the Reconstruct RPC of ReconstructionService.
//
// Use Reconstruct() directly instead of the generic Process().
// Process() is provided only to satisfy InferenceClientBase and always
// returns FailedPreconditionError.
//
// Reconstruct() parameters:
//   client_id   : caller-supplied identifier
//   image_data  : original image bytes
//   mask_data   : raw mask bytes from Sam3SegClient (mask_height * mask_width,
//                 uint8, 0=background 1=foreground)
//   mask_width  : mask column count
//   mask_height : mask row count
//
// InferenceResult on success:
//   output_data : GLB file bytes
//   json_data   : transform JSON bytes
class Sam3dObjClient : public inference::InferenceClientBase {
 public:
  Sam3dObjClient();
  ~Sam3dObjClient() override = default;

  absl::Status ConnectToAddress(
      const std::string& address_with_port) override;

  // Not applicable for Reconstruct; always returns FailedPreconditionError.
  std::future<absl::StatusOr<inference::InferenceResult>> Process(
      const std::string& client_id,
      const std::string& input_data,
      const float* params, size_t params_size) override;

  // Issue an async Reconstruct RPC.
  std::future<absl::StatusOr<inference::InferenceResult>> Reconstruct(
      const std::string& client_id,
      const std::string& image_data,
      const std::string& mask_data,
      int32_t mask_width,
      int32_t mask_height);

 private:
  absl::Status OnConnect() override;

  std::unique_ptr<spatialblend::scan::v1::ReconstructionService::Stub> stub_;
};

}  // namespace spatialblend::distributed

#endif  // SESSION_INFERENCE_DISTRIBUTED_SAM3D_OBJ_CLIENT_H_
