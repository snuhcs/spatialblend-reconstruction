// Copyright (c) 2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SESSION_INFERENCE_DISTRIBUTED_SAM3_SEG_CLIENT_H_
#define SESSION_INFERENCE_DISTRIBUTED_SAM3_SEG_CLIENT_H_

#include <cstddef>
#include <future>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "session/inference/inference_client.h"
#include "session/inference/distributed/proto/distributed.grpc.pb.h"
#include "session/inference/distributed/proto/distributed.pb.h"

namespace spatialblend::distributed {

struct Sam3SegClientCall : public GrpcClientCallBase {
  using GrpcClientCallBase::GrpcClientCallBase;
  ~Sam3SegClientCall() override = default;
};

// Async gRPC client for the Segment RPC of ReconstructionService.
//
// Process() parameters:
//   input_data  : image bytes
//   params      : flat [x1, y1, 0, x2, y2, 0, ...] (xyz triplets, z ignored)
//   params_size : num_points * 3
//
// InferenceResult on success:
//   output_data  : raw mask bytes (mask_height * mask_width, uint8, 0 or 1)
//   mask_width   : mask column count
//   mask_height  : mask row count
class Sam3SegClient : public inference::InferenceClientBase {
 public:
  Sam3SegClient();
  ~Sam3SegClient() override = default;

  absl::Status ConnectToAddress(
      const std::string& address_with_port) override;

  std::future<absl::StatusOr<inference::InferenceResult>> Process(
      const std::string& client_id,
      const std::string& input_data,
      const float* params, size_t params_size) override;

 private:
  absl::Status OnConnect() override;

  std::unique_ptr<spatialblend::scan::v1::ReconstructionService::Stub> stub_;
};

}  // namespace spatialblend::distributed

#endif  // SESSION_INFERENCE_DISTRIBUTED_SAM3_SEG_CLIENT_H_
