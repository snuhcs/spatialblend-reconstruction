// Copyright (c) 2026 Seoul National University
// SPDX-License-Identifier: MIT

// TLS integration test for the native inference client.
//
// Starts an in-process gRPC server with SslServerCredentials using the
// self-signed "localhost" certificate generated at test time and checks that:
//   * kTls + SPATIALBLEND_TLS_ROOT_CERTS_FILE=<that cert> completes an RPC;
//   * kTls with default roots rejects the self-signed certificate;
//   * kTls rejects a host-name mismatch (certificate is for "localhost" only,
//     the client connects to 127.0.0.1) -- there is no target-name override;
//   * plaintext against the TLS port fails, and plaintext against a plaintext
//     server still works (default transport unchanged).
// The localhost certificate and key are generated in-process at test time
// (P-256, 1 day validity, SAN DNS:localhost) with the BoringSSL that gRPC links.
// Nothing key-like is checked in.

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>

#include <openssl/bio.h>
#include <openssl/ec.h>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/x509.h>
#include <openssl/x509v3.h>

#include "gtest/gtest.h"
#include "session/inference/inference_client.h"
#include "session/inference/sam3d/proto/sam3d.grpc.pb.h"
#include "session/inference/sam3d/sam3d_client.h"

namespace spatialblend::sam3d {
namespace {

using inference::InferenceClientBase;
using spatialblend::scan::v1::InferenceRequest;
using spatialblend::scan::v1::InferenceResponse;
using spatialblend::scan::v1::OneShotReconstructionService;
using TransportSecurity = InferenceClientBase::TransportSecurity;

constexpr char kRootsEnv[] = "SPATIALBLEND_TLS_ROOT_CERTS_FILE";

void SetEnv(const char* name, const char* value) {
#ifdef _WIN32
  _putenv_s(name, value ? value : "");
#else
  if (value) {
    setenv(name, value, 1);
  } else {
    unsetenv(name);
  }
#endif
}

class EchoService final : public OneShotReconstructionService::Service {
 public:
  grpc::Status ProcessImage(grpc::ServerContext*, const InferenceRequest* req,
                            InferenceResponse* resp) override {
    resp->set_success(true);
    resp->set_message("ok:" + req->client_id());
    return grpc::Status::OK;
  }
};

template <typename T, void (*Free)(T*)>
struct Deleter {
  void operator()(T* p) const { if (p) Free(p); }
};
using BioPtr = std::unique_ptr<BIO, Deleter<BIO, BIO_vfree>>;
using PkeyPtr = std::unique_ptr<EVP_PKEY, Deleter<EVP_PKEY, EVP_PKEY_free>>;
using X509Ptr = std::unique_ptr<X509, Deleter<X509, X509_free>>;

std::string BioToString(BIO* bio) {
  char* data = nullptr;
  long n = BIO_get_mem_data(bio, &data);
  return std::string(data, static_cast<size_t>(n));
}

// Generates a self-signed P-256 certificate for CN/SAN "localhost", valid for
// one day. Returns false (with a message) on any OpenSSL failure.
bool GenerateSelfSignedLocalhost(std::string* cert_pem, std::string* key_pem,
                                 std::string* error) {
  auto fail = [&](const char* what) {
    *error = std::string("self-signed cert generation failed: ") + what;
    return false;
  };
  EC_KEY* ec = EC_KEY_new_by_curve_name(NID_X9_62_prime256v1);
  if (!ec || !EC_KEY_generate_key(ec)) {
    if (ec) EC_KEY_free(ec);
    return fail("EC key");
  }
  PkeyPtr pkey(EVP_PKEY_new());
  if (!pkey || !EVP_PKEY_assign_EC_KEY(pkey.get(), ec)) {
    EC_KEY_free(ec);
    return fail("EVP_PKEY");
  }
  X509Ptr x(X509_new());
  if (!x) return fail("X509_new");
  X509_set_version(x.get(), 2);
  ASN1_INTEGER_set(X509_get_serialNumber(x.get()), 1);
  X509_gmtime_adj(X509_getm_notBefore(x.get()), -3600);
  X509_gmtime_adj(X509_getm_notAfter(x.get()), 24 * 3600);
  X509_set_pubkey(x.get(), pkey.get());
  X509_NAME* name = X509_get_subject_name(x.get());
  X509_NAME_add_entry_by_txt(name, "CN", MBSTRING_ASC,
                             reinterpret_cast<const unsigned char*>("localhost"),
                             -1, -1, 0);
  X509_set_issuer_name(x.get(), name);
  // SAN DNS:localhost
  GENERAL_NAMES* sans = sk_GENERAL_NAME_new_null();
  GENERAL_NAME* dns = GENERAL_NAME_new();
  ASN1_IA5STRING* ia5 = ASN1_IA5STRING_new();
  if (!sans || !dns || !ia5 ||
      !ASN1_STRING_set(ia5, "localhost", 9)) {
    return fail("SAN");
  }
  GENERAL_NAME_set0_value(dns, GEN_DNS, ia5);
  sk_GENERAL_NAME_push(sans, dns);
  int ok = X509_add1_ext_i2d(x.get(), NID_subject_alt_name, sans, 0, 0);
  GENERAL_NAMES_free(sans);
  if (ok != 1) return fail("SAN extension");
  // basicConstraints CA:TRUE (self-signed trust anchor)
  BASIC_CONSTRAINTS* bc = BASIC_CONSTRAINTS_new();
  bc->ca = 0xff;
  ok = X509_add1_ext_i2d(x.get(), NID_basic_constraints, bc, 1, 0);
  BASIC_CONSTRAINTS_free(bc);
  if (ok != 1) return fail("basicConstraints");
  if (!X509_sign(x.get(), pkey.get(), EVP_sha256())) return fail("sign");

  BioPtr cbio(BIO_new(BIO_s_mem()));
  BioPtr kbio(BIO_new(BIO_s_mem()));
  if (!PEM_write_bio_X509(cbio.get(), x.get()) ||
      !PEM_write_bio_PrivateKey(kbio.get(), pkey.get(), nullptr, nullptr, 0,
                                nullptr, nullptr)) {
    return fail("PEM encode");
  }
  *cert_pem = BioToString(cbio.get());
  *key_pem = BioToString(kbio.get());
  return true;
}

void WriteFile(const std::string& path, const std::string& data) {
  std::ofstream f(path, std::ios::binary);
  f << data;
}

class TlsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::string err;
    ASSERT_TRUE(GenerateSelfSignedLocalhost(&cert_pem_, &key_pem_, &err)) << err;
    const char* tmp = std::getenv("TEST_TMPDIR");
    cert_path_ = std::string(tmp ? tmp : ".") + "/localhost_test_root.pem";
    WriteFile(cert_path_, cert_pem_);
    SetEnv(kRootsEnv, nullptr);
  }
  void TearDown() override { SetEnv(kRootsEnv, nullptr); }

  // Returns the bound port, or 0.
  int StartServer(bool tls) {
    grpc::ServerBuilder builder;
    std::shared_ptr<grpc::ServerCredentials> creds;
    if (tls) {
      grpc::SslServerCredentialsOptions opts;
      opts.pem_key_cert_pairs.push_back({key_pem_, cert_pem_});
      creds = grpc::SslServerCredentials(opts);
    } else {
      creds = grpc::InsecureServerCredentials();
    }
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0", creds, &port);
    builder.RegisterService(&service_);
    server_ = builder.BuildAndStart();
    return server_ ? port : 0;
  }

  // Runs one ProcessImage; returns true only if the RPC reached the service.
  bool Call(TransportSecurity security, const std::string& host, int port) {
    auto client = std::make_shared<Sam3dClient>();
    client->SetTransportSecurity(security);
    absl::Status st = client->ConnectToAddress(host + ":" + std::to_string(port));
    if (!st.ok()) {
      ADD_FAILURE() << "connect: " << st;
      return false;
    }
    auto fut = client->Process("tls-test", "x", nullptr, 0);
    if (fut.wait_for(std::chrono::seconds(60)) != std::future_status::ready) {
      ADD_FAILURE() << "RPC timed out";
      return false;
    }
    auto result = fut.get();
    (void)client->Disconnect();
    return result.ok() && result->success;
  }

  EchoService service_;
  std::unique_ptr<grpc::Server> server_;
  std::string cert_path_, cert_pem_, key_pem_;
};

TEST_F(TlsTest, CredentialsFactoryIsDistinctPerTransport) {
  auto insecure =
      InferenceClientBase::CreateChannelCredentials(TransportSecurity::kInsecure);
  auto tls = InferenceClientBase::CreateChannelCredentials(TransportSecurity::kTls);
  ASSERT_TRUE(insecure.ok());
  ASSERT_TRUE(tls.ok());
  EXPECT_NE(insecure->get(), tls->get());

  SetEnv(kRootsEnv, "/nonexistent/roots.pem");
  auto bad = InferenceClientBase::CreateChannelCredentials(TransportSecurity::kTls);
  EXPECT_TRUE(absl::IsInvalidArgument(bad.status())) << bad.status();
  // The override is a TLS-only option.
  EXPECT_TRUE(InferenceClientBase::CreateChannelCredentials(
                  TransportSecurity::kInsecure)
                  .ok());
}

TEST_F(TlsTest, TlsWithSuppliedRootSucceeds) {
  const int port = StartServer(/*tls=*/true);
  ASSERT_GT(port, 0);
  SetEnv(kRootsEnv, cert_path_.c_str());
  EXPECT_TRUE(Call(TransportSecurity::kTls, "localhost", port));
}

TEST_F(TlsTest, TlsRejectsUntrustedSelfSignedCertificate) {
  const int port = StartServer(/*tls=*/true);
  ASSERT_GT(port, 0);
  EXPECT_FALSE(Call(TransportSecurity::kTls, "localhost", port));
}

TEST_F(TlsTest, TlsVerifiesHostName) {
  const int port = StartServer(/*tls=*/true);
  ASSERT_GT(port, 0);
  SetEnv(kRootsEnv, cert_path_.c_str());
  // Trusted root, but the certificate is for "localhost", not 127.0.0.1.
  EXPECT_FALSE(Call(TransportSecurity::kTls, "127.0.0.1", port));
}

TEST_F(TlsTest, PlaintextClientCannotTalkToTlsServer) {
  const int port = StartServer(/*tls=*/true);
  ASSERT_GT(port, 0);
  EXPECT_FALSE(Call(TransportSecurity::kInsecure, "127.0.0.1", port));
}

TEST_F(TlsTest, PlaintextStillWorksByDefault) {
  const int port = StartServer(/*tls=*/false);
  ASSERT_GT(port, 0);
  EXPECT_TRUE(Call(TransportSecurity::kInsecure, "127.0.0.1", port));
}

}  // namespace
}  // namespace spatialblend::sam3d
