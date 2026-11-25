# Copyright (c) 2026 Seoul National University
# SPDX-License-Identifier: MIT

load("@grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")

package(
    default_visibility = ["//visibility:public"],
)

exports_files([
    "sam3d.proto",
])

proto_library(
    name = "sam3d_proto",
    srcs = ["sam3d.proto"],
)

cc_proto_library(
    name = "sam3d_cc_proto",
    deps = [":sam3d_proto"],
)

cc_grpc_library(
    name = "sam3d_cc_grpc",
    srcs = [":sam3d_proto"],
    grpc_only = True,
    deps = [
        ":sam3d_cc_proto",
        "@grpc//:grpc++_reflection",
    ],
)
