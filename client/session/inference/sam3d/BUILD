# Copyright (c) 2026 Seoul National University
# SPDX-License-Identifier: MIT

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "sam3d_client",
    srcs = ["sam3d_client.cc"],
    hdrs = ["sam3d_client.h"],
    deps = [
        "//session/inference:inference_client",
        "//session/inference/sam3d/proto:sam3d_cc_grpc",
        "//session/network:proto_util",
        "//session/util:util",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@grpc//:grpc++",
    ],
)

filegroup(
    name = "test_data",
    srcs = glob(["test_data/**"]),
    visibility = ["//visibility:public"],
)

load("//session:session.bzl", "session_cc_test")

session_cc_test(
    name = "sam3d_remote_test",
    srcs = ["sam3d_remote_test.cc"],
    data = glob(["test_data/**"]),
    tags = ["manual"],
    deps = [
        ":sam3d_client",
        "//session/inference:inference_client",
        "@googletest//:gtest_main",
    ],
)

# TLS integration test: generates a self-signed localhost certificate at test
# time, serves it in-process and connects with TLS via
# SPATIALBLEND_TLS_ROOT_CERTS_FILE. Runs offline.
session_cc_test(
    name = "sam3d_tls_test",
    srcs = ["sam3d_tls_test.cc"],
    deps = [
        ":sam3d_client",
        "//session/inference:inference_client",
        "//session/inference/sam3d/proto:sam3d_cc_grpc",
        "@boringssl//:crypto",
        "@googletest//:gtest_main",
        "@grpc//:grpc++",
    ],
)
