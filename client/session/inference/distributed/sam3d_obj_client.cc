// Copyright (c) 2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "session/inference/distributed/sam3d_obj_client.h"

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
using spatialblend::scan::v1::ReconstructRequest;
using spatialblend::scan::v1::ReconstructResponse;
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

struct ReconstructCall final : public Sam3dObjClientCall {
  explicit ReconstructCall(
      std::shared_ptr<Sam3dObjClient> client,
      std::shared_ptr<std::promise<absl::StatusOr<inference::InferenceResult>>>
          promise)
      : Sam3dObjClientCall(std::move(client)), promise_(std::move(promise)) {}

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

  ReconstructRequest request_;
  ReconstructResponse response_;
  std::unique_ptr<grpc::ClientAsyncResponseReader<ReconstructResponse>> rpc_;
  std::shared_ptr<std::promise<absl::StatusOr<inference::InferenceResult>>>
      promise_;
};

Sam3dObjClient::Sam3dObjClient() = default;

absl::Status Sam3dObjClient::ConnectToAddress(
    const std::string& address_with_port) {
  std::string host;
  int port = 0;
  RETURN_IF_ERROR(ParseHostPort(address_with_port, &host, &port));

  grpc::ChannelArguments args;
  args.SetMaxReceiveMessageSize(kMaxMessageBytes);
  args.SetMaxSendMessageSize(kMaxMessageBytes);
  return Connect(host, port, args);
}

absl::Status Sam3dObjClient::OnConnect() {
  stub_ = ReconstructionService::NewStub(channel_);
  if (!stub_) {
    return absl::InternalError(
        "Failed to create ReconstructionService stub (Reconstruct)");
  }
  return absl::OkStatus();
}

std::future<absl::StatusOr<inference::InferenceResult>>
Sam3dObjClient::Process(const std::string& /*client_id*/,
                        const std::string& /*input_data*/,
                        const float* /*params*/, size_t /*params_size*/) {
  return util::MakeReadyFuture(absl::StatusOr<inference::InferenceResult>(
      absl::FailedPreconditionError(
          "Sam3dObjClient does not support generic Process(); "
          "use Reconstruct() instead")));
}

std::future<absl::StatusOr<inference::InferenceResult>>
Sam3dObjClient::Reconstruct(const std::string& client_id,
                             const std::string& image_data,
                             const std::string& mask_data, int32_t mask_width,
                             int32_t mask_height) {
  if (!stub_ || !IsConnected()) {
    return util::MakeReadyFuture(absl::StatusOr<inference::InferenceResult>(
        absl::FailedPreconditionError("Sam3dObjClient is not connected")));
  }

  auto promise = std::make_shared<
      std::promise<absl::StatusOr<inference::InferenceResult>>>();
  std::future<absl::StatusOr<inference::InferenceResult>> future =
      promise->get_future();

  auto self = std::static_pointer_cast<Sam3dObjClient>(shared_from_this());
  auto* call = new ReconstructCall(self, promise);

  call->request_.set_client_id(client_id);
  call->request_.set_image_data(image_data);
  call->request_.set_mask_data(mask_data);
  call->request_.set_mask_width(mask_width);
  call->request_.set_mask_height(mask_height);

  call->context_.set_deadline(std::chrono::system_clock::now() +
                              std::chrono::minutes(10));

  call->rpc_ = stub_->PrepareAsyncReconstruct(&call->context_, call->request_,
                                              cq_.get());
  call->rpc_->StartCall();
  call->rpc_->Finish(&call->response_, &call->status_, (void*)call);

  return future;
}

}  // namespace spatialblend::distributed
