// Copyright (c) 2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SESSION_INFERENCE_SAM3D_SAM3D_CLIENT_H_
#define SESSION_INFERENCE_SAM3D_SAM3D_CLIENT_H_

#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "session/inference/inference_client.h"
#include "session/inference/sam3d/proto/sam3d.grpc.pb.h"
#include "session/inference/sam3d/proto/sam3d.pb.h"

namespace spatialblend::sam3d {

// Marker type for CQ tags used by Sam3D RPCs.
struct Sam3dClientCall : public GrpcClientCallBase {
  using GrpcClientCallBase::GrpcClientCallBase;
  ~Sam3dClientCall() override = default;
};

// Thin async client for the external Sam3D gRPC server.
class Sam3dClient : public inference::InferenceClientBase {
 public:
  Sam3dClient();
  ~Sam3dClient() override = default;

  // --- InferenceClient interface ---
  absl::Status ConnectToAddress(
      const std::string& address_with_port) override;

  // Generic process: params are flat xyz triplets [x1,y1,z1, x2,y2,z2, ...]
  std::future<absl::StatusOr<inference::InferenceResult>> Process(
      const std::string& client_id,
      const std::string& input_data,
      const float* params, size_t params_size) override;

 private:
  absl::Status OnConnect() override;

  std::unique_ptr<spatialblend::scan::v1::OneShotReconstructionService::Stub> stub_;
};

}  // namespace spatialblend::sam3d

#endif  // SESSION_INFERENCE_SAM3D_SAM3D_CLIENT_H_
