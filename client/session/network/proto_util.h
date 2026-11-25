// Copyright (c) 2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SESSION_NETWORK_PROTO_UTIL_H_
#define SESSION_NETWORK_PROTO_UTIL_H_

#include <grpcpp/grpcpp.h>

#include "absl/status/status.h"

namespace spatialblend {
namespace proto_util {

// Convert between gRPC and Abseil status
absl::Status GrpcStatusToAbsl(const grpc::Status& status);
grpc::Status AbslStatusToGrpc(const absl::Status& status);

}  // namespace proto_util
}  // namespace spatialblend

#endif /* SESSION_NETWORK_PROTO_UTIL_H_ */
