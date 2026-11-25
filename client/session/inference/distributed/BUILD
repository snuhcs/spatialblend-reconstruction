# Copyright (c) 2026 Seoul National University
# SPDX-License-Identifier: MIT

package(
    default_visibility = ["//visibility:public"],
)

load("//session:session.bzl", "session_cc_test")

cc_library(
    name = "sam3_seg_client",
    srcs = ["sam3_seg_client.cc"],
    hdrs = ["sam3_seg_client.h"],
    deps = [
        "//session/inference:inference_client",
        "//session/inference/distributed/proto:distributed_cc_grpc",
        "//session/network:proto_util",
        "//session/util:util",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@grpc//:grpc++",
    ],
)

cc_library(
    name = "sam3d_obj_client",
    srcs = ["sam3d_obj_client.cc"],
    hdrs = ["sam3d_obj_client.h"],
    deps = [
        "//session/inference:inference_client",
        "//session/inference/distributed/proto:distributed_cc_grpc",
        "//session/network:proto_util",
        "//session/util:util",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@grpc//:grpc++",
    ],
)

session_cc_test(
    name = "distributed_test",
    srcs = ["distributed_test.cc"],
    data = ["//session/inference/sam3d:test_data"],
    tags = ["manual"],
    deps = [
        ":sam3_seg_client",
        ":sam3d_obj_client",
        "//session/inference:inference_client",
        "//third_party/stb:stb_image",
        "@googletest//:gtest_main",
    ],
)
