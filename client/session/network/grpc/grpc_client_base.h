// Copyright (c) 2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SESSION_NETWORK_GRPC_CLIENT_BASE_H_
#define SESSION_NETWORK_GRPC_CLIENT_BASE_H_

#include <grpcpp/grpcpp.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_set>

#include "absl/status/status.h"
#include "session/network/util/network_util.h"
#include "session/util/logger.h"
#include "session/util/util.h"

namespace spatialblend {

struct GrpcClientCallBase;

namespace internal {
// Interface for managing pending gRPC calls
class IGrpcClient : public std::enable_shared_from_this<IGrpcClient> {
 public:
  virtual ~IGrpcClient() = default;

  virtual void AddPendingCall(GrpcClientCallBase* call) = 0;
  virtual void RemovePendingCall(GrpcClientCallBase* call) = 0;
};
}  // namespace internal

struct GrpcClientCallBase {
  GrpcClientCallBase(std::shared_ptr<internal::IGrpcClient> client)
      : client_(client) {
    if (auto client_ptr = client_.lock()) {
      client_ptr->AddPendingCall(this);
    }
  }
  virtual ~GrpcClientCallBase() {
    if (auto client_ptr = client_.lock()) {
      client_ptr->RemovePendingCall(this);
    }
  }
  virtual void Proceed(bool ok) = 0;

  grpc::ClientContext context_;
  grpc::Status status_;
  std::weak_ptr<internal::IGrpcClient> client_;
};

}  // namespace spatialblend

#endif  // SESSION_NETWORK_GRPC_CLIENT_BASE_H_
