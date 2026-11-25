// Copyright (c) 2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "session/inference/distributed/sam3_seg_client.h"

#include <chrono>
#include <memory>
#include <string>
#include <utility>

#include "absl/strings/str_format.h"
#include "session/network/proto_util.h"
#include "session/util/util.h"

namespace spatialblend::distributed {
namespace {

// Generated from package spatialblend.scan.v1.
using spatialblend::scan::v1::SegmentRequest;
using spatialblend::scan::v1::SegmentResponse;
using spatialblend::scan::v1::SegPoint;
using spatialblend::scan::v1::ReconstructionService;

constexpr int kMaxMessageBytes = 100 * 1024 * 1024;

absl::Status ParseHostPort(const std::string& address_with_port,
                           std::string* host, int* port) {
  auto pos = address_with_port.rfind(':');
  if (pos == std::string::npos || pos == 0 ||
      pos + 1 >= address_with_port.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Invalid address (expected host:port): %s", address_with_port));
  }
  *host = address_with_port.substr(0, pos);
  const std::string port_str = address_with_port.substr(pos + 1);
  try {
    *port = std::stoi(port_str);
  } catch (...) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Invalid port in address: %s", address_with_port));
  }
  return absl::OkStatus();
}

}  // namespace

struct SegmentCall final : public Sam3SegClientCall {
  explicit SegmentCall(
      std::shared_ptr<Sam3SegClient> client,
      std::shared_ptr<std::promise<absl::StatusOr<inference::InferenceResult>>>
          promise)
      : Sam3SegClientCall(std::move(client)), promise_(std::move(promise)) {}

  void Proceed(bool /*ok*/) override {
    absl::Status st = proto_util::GrpcStatusToAbsl(status_);
    if (!st.ok()) {
      if (promise_) promise_->set_value(st);
      delete this;
      return;
    }

    inference::InferenceResult result;
    result.success = response_.success();
    result.message = response_.message();
    result.output_data = response_.mask_data();  // raw mask bytes
    result.mask_width = response_.mask_width();
    result.mask_height = response_.mask_height();
    if (promise_) promise_->set_value(std::move(result));
    delete this;
  }

  SegmentRequest request_;
  SegmentResponse response_;
  std::unique_ptr<grpc::ClientAsyncResponseReader<SegmentResponse>> rpc_;
  std::shared_ptr<std::promise<absl::StatusOr<inference::InferenceResult>>>
      promise_;
};

Sam3SegClient::Sam3SegClient() = default;

absl::Status Sam3SegClient::ConnectToAddress(
    const std::string& address_with_port) {
  std::string host;
  int port = 0;
  RETURN_IF_ERROR(ParseHostPort(address_with_port, &host, &port));

  grpc::ChannelArguments args;
  args.SetMaxReceiveMessageSize(kMaxMessageBytes);
  args.SetMaxSendMessageSize(kMaxMessageBytes);
  return Connect(host, port, args);
}

absl::Status Sam3SegClient::OnConnect() {
  stub_ = ReconstructionService::NewStub(channel_);
  if (!stub_) {
    return absl::InternalError(
        "Failed to create ReconstructionService stub (Segment)");
  }
  return absl::OkStatus();
}

std::future<absl::StatusOr<inference::InferenceResult>> Sam3SegClient::Process(
    const std::string& client_id, const std::string& input_data,
    const float* params, size_t params_size) {
  if (!stub_ || !IsConnected()) {
    return util::MakeReadyFuture(absl::StatusOr<inference::InferenceResult>(
        absl::FailedPreconditionError("Sam3SegClient is not connected")));
  }

  auto promise = std::make_shared<
      std::promise<absl::StatusOr<inference::InferenceResult>>>();
  std::future<absl::StatusOr<inference::InferenceResult>> future =
      promise->get_future();

  auto self = std::static_pointer_cast<Sam3SegClient>(shared_from_this());
  auto* call = new SegmentCall(self, promise);

  call->request_.set_client_id(client_id);
  call->request_.set_image_data(input_data);

  // params: flat [x1, y1, z1, x2, y2, z2, ...] — z is ignored by the server
  if (params && params_size >= 3) {
    size_t num_points = params_size / 3;
    for (size_t i = 0; i < num_points; ++i) {
      SegPoint* p = call->request_.add_points();
      p->set_x(params[i * 3]);
      p->set_y(params[i * 3 + 1]);
      p->set_z(params[i * 3 + 2]);
    }
  }

  call->context_.set_deadline(std::chrono::system_clock::now() +
                              std::chrono::minutes(5));

  call->rpc_ = stub_->PrepareAsyncSegment(&call->context_, call->request_,
                                          cq_.get());
  call->rpc_->StartCall();
  call->rpc_->Finish(&call->response_, &call->status_, (void*)call);

  return future;
}

}  // namespace spatialblend::distributed
