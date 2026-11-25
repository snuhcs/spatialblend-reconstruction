// Copyright (c) 2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "session/inference/inference_client.h"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iterator>

#include "absl/strings/str_format.h"
#include "session/network/util/network_util.h"
#include "session/util/logger.h"
#include "session/util/util.h"

namespace spatialblend::inference {
namespace {

// RFC 1035-style host name: dot-separated labels of [A-Za-z0-9-], 1-63 chars
// each, not starting or ending with '-', 253 chars at most.
bool IsWellFormedDnsName(const std::string& host) {
  if (host.empty() || host.size() > 253) return false;
  size_t label_len = 0;
  char prev = '.';
  for (char c : host) {
    if (c == '.') {
      if (label_len == 0 || prev == '-') return false;
      label_len = 0;
    } else if (std::isalnum(static_cast<unsigned char>(c)) || c == '-') {
      if (label_len == 0 && c == '-') return false;
      if (++label_len > 63) return false;
    } else {
      return false;
    }
    prev = c;
  }
  return label_len != 0 && prev != '-';
}

}  // namespace

InferenceClientBase::~InferenceClientBase() { LOG_IF_ERROR(Disconnect()); }

absl::Status InferenceClientBase::Connect(const std::string& address,
                                          int port) {
  // CreateChannel() is CreateCustomChannel() with default arguments.
  return Connect(address, port, grpc::ChannelArguments());
}

absl::Status InferenceClientBase::Connect(
    const std::string& address, int port,
    const grpc::ChannelArguments& channel_args) {
  if (IsConnected()) {
    return absl::AlreadyExistsError("Already connected");
  }

  const bool use_tls = transport_security_ == TransportSecurity::kTls;
  // With TLS the address is typically a DNS name such as
  // "scan-host.example", possibly with more than one dot, which the plaintext
  // validator rejects; accept any well-formed DNS name and let gRPC resolve it.
  const bool address_ok =
      (use_tls && IsWellFormedDnsName(address)) ||
      network::util::IsValidHostAddress(address);
  if (!address_ok) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Invalid or unreachable host address %s", address.c_str()));
  }

  if (!network::util::IsValidPort(port)) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Invalid port %d", port));
  }

  auto creds = CreateChannelCredentials(transport_security_);
  if (!creds.ok()) {
    return creds.status();
  }

  // Under TLS force the DNS resolver: a bare "unix:443" or "ipv4:443" would
  // otherwise be parsed by gRPC as a URI scheme rather than a host name.
  const std::string target =
      (use_tls ? "dns:///" : "") + address + ":" + std::to_string(port);
  channel_ = grpc::CreateCustomChannel(target, *creds, channel_args);
  cq_ = std::make_unique<grpc::CompletionQueue>();

  RETURN_IF_ERROR(OnConnect());

  shutdown_ = false;
  connected_ = true;
  async_thread_ = std::thread(&InferenceClientBase::HandleRpcs, this);

  return absl::OkStatus();
}

absl::StatusOr<std::shared_ptr<grpc::ChannelCredentials>>
InferenceClientBase::CreateChannelCredentials(TransportSecurity security) {
  if (security == TransportSecurity::kInsecure) {
    return grpc::InsecureChannelCredentials();
  }

  grpc::SslCredentialsOptions options;  // empty roots: gRPC default discovery
  const char* roots_path = std::getenv(kTlsRootCertsFileEnvVar);
  if (roots_path != nullptr && roots_path[0] != '\0') {
    std::ifstream file(roots_path, std::ios::binary);
    std::string pem((std::istreambuf_iterator<char>(file)),
                    std::istreambuf_iterator<char>());
    if (!file.is_open() || pem.empty()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "%s is set but %s is unreadable or empty", kTlsRootCertsFileEnvVar,
          roots_path));
    }
    options.pem_root_certs = std::move(pem);
  }
  return grpc::SslCredentials(options);
}

absl::Status InferenceClientBase::Disconnect() {
  shutdown_ = true;
  connected_ = false;

  {
    std::lock_guard<std::mutex> lock(pending_calls_mutex_);
    for (auto* call : pending_calls_) {
      call->context_.TryCancel();
    }
    pending_calls_.clear();
  }

  if (cq_) {
    cq_->Shutdown();

    void* ignored_tag;
    bool ignored_ok;
    while (cq_->Next(&ignored_tag, &ignored_ok)) {
    }
  }

  if (async_thread_.joinable()) {
    async_thread_.join();
  }

  channel_.reset();
  cq_.reset();
  return absl::OkStatus();
}

bool InferenceClientBase::IsConnected() const {
  return channel_ != nullptr;
}

void InferenceClientBase::HandleRpcs() {
  void* got_tag;
  bool ok = false;

  while (cq_->Next(&got_tag, &ok)) {
    if (!ok || shutdown_.load()) {
      break;
    }
    static_cast<GrpcClientCallBase*>(got_tag)->Proceed(ok);
  }
}

void InferenceClientBase::AddPendingCall(GrpcClientCallBase* call) {
  std::lock_guard<std::mutex> lock(pending_calls_mutex_);
  pending_calls_.insert(call);
}

void InferenceClientBase::RemovePendingCall(GrpcClientCallBase* call) {
  std::lock_guard<std::mutex> lock(pending_calls_mutex_);
  pending_calls_.erase(call);
}

}  // namespace spatialblend::inference
