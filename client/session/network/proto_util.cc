// Copyright (c) 2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "session/network/proto_util.h"

namespace spatialblend {
namespace proto_util {

absl::Status GrpcStatusToAbsl(const grpc::Status& status) {
  return absl::Status(static_cast<absl::StatusCode>(status.error_code()),
                      status.error_message());
}

grpc::Status AbslStatusToGrpc(const absl::Status& status) {
  return grpc::Status(static_cast<grpc::StatusCode>(status.code()),
                      status.ToString());
}

}  // namespace proto_util
}  // namespace spatialblend
