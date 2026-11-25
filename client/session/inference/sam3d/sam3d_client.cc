// Copyright (c) 2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "session/inference/sam3d/sam3d_client.h"

#include <chrono>
#include <memory>
#include <string>
#include <utility>

#include "absl/strings/str_format.h"
#include "session/network/proto_util.h"
#include "session/util/util.h"

namespace spatialblend::sam3d {
namespace {

// Generated from package spatialblend.scan.v1.
using spatialblend::scan::v1::InferenceRequest;
using spatialblend::scan::v1::InferenceResponse;
using spatialblend::scan::v1::Point;
using spatialblend::scan::v1::OneShotReconstructionService;

constexpr int kMaxMessageBytes = 100 * 1024 * 1024;  // match python client intent

absl::Status ParseHostPort(const std::string& address_with_port,
                           std::string* host, int* port) {
  auto pos = address_with_port.rfind(':');
  if (pos == std::string::npos || pos == 0 ||
      pos + 1 >= address_with_port.size()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Invalid address (expected host:port): %s",
                        address_with_port));
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

struct ProcessImageCall final : public Sam3dClientCall {
  explicit ProcessImageCall(
      std::shared_ptr<Sam3dClient> client,
      std::shared_ptr<std::promise<absl::StatusOr<inference::InferenceResult>>> promise)
      : Sam3dClientCall(std::move(client)), promise_(std::move(promise)) {}

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
    result.output_data = response_.glb_file();
    result.json_data = response_.json_file();
    if (promise_) promise_->set_value(std::move(result));
    delete this;
  }

  InferenceRequest request_;
  InferenceResponse response_;
  std::unique_ptr<grpc::ClientAsyncResponseReader<InferenceResponse>> rpc_;
  std::shared_ptr<std::promise<absl::StatusOr<inference::InferenceResult>>> promise_;
};

Sam3dClient::Sam3dClient() = default;

absl::Status Sam3dClient::ConnectToAddress(const std::string& address_with_port) {
  std::string host;
  int port = 0;
  RETURN_IF_ERROR(ParseHostPort(address_with_port, &host, &port));

  grpc::ChannelArguments args;
  args.SetMaxReceiveMessageSize(kMaxMessageBytes);
  args.SetMaxSendMessageSize(kMaxMessageBytes);
  return Connect(host, port, args);
}

absl::Status Sam3dClient::OnConnect() {
  stub_ = OneShotReconstructionService::NewStub(channel_);
  if (!stub_) {
    return absl::InternalError("Failed to create OneShotReconstructionService stub");
  }
  return absl::OkStatus();
}

std::future<absl::StatusOr<inference::InferenceResult>> Sam3dClient::Process(
    const std::string& client_id, const std::string& input_data,
    const float* params, size_t params_size) {
  if (!stub_ || !IsConnected()) {
    return util::MakeReadyFuture(absl::StatusOr<inference::InferenceResult>(
        absl::FailedPreconditionError("Sam3dClient is not connected")));
  }

  auto promise = std::make_shared<std::promise<absl::StatusOr<inference::InferenceResult>>>();
  std::future<absl::StatusOr<inference::InferenceResult>> future = promise->get_future();

  auto self = std::static_pointer_cast<Sam3dClient>(shared_from_this());
  auto* call = new ProcessImageCall(self, promise);

  call->request_.set_client_id(client_id);
  call->request_.set_image_data(input_data);

  // Convert flat float params to proto Point: [x1,y1,z1, x2,y2,z2, ...]
  if (params && params_size >= 3) {
    size_t num_points = params_size / 3;
    for (size_t i = 0; i < num_points; ++i) {
      Point* p = call->request_.add_points();
      p->set_x(params[i * 3]);
      p->set_y(params[i * 3 + 1]);
      p->set_z(params[i * 3 + 2]);
    }
  }

  call->context_.set_deadline(std::chrono::system_clock::now() +
                              std::chrono::minutes(10));

  call->rpc_ = stub_->PrepareAsyncProcessImage(&call->context_, call->request_,
                                               cq_.get());
  call->rpc_->StartCall();
  call->rpc_->Finish(&call->response_, &call->status_, (void*)call);

  return future;
}

}  // namespace spatialblend::sam3d
