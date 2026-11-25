# Copyright (c) 2026 Seoul National University
# SPDX-License-Identifier: MIT

load("@grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")

package(
    default_visibility = ["//visibility:public"],
)

exports_files(["scan_proto.BUILD.bazel"])

# The proto is the repository's top-level proto/distributed.proto (@scan_proto, MODULE.bazel): one source of truth
# for this client and the Python server in ../server. It is copied into this package at build time (never checked in
# here) so the generated headers stay at session/inference/distributed/proto/distributed{,.grpc}.pb.h.
genrule(
    name = "distributed_proto_src",
    srcs = ["@scan_proto//:distributed.proto"],
    outs = ["distributed.proto"],
    cmd = "cp $< $@",
)

proto_library(
    name = "distributed_proto",
    srcs = [":distributed.proto"],
)

cc_proto_library(
    name = "distributed_cc_proto",
    deps = [":distributed_proto"],
)

cc_grpc_library(
    name = "distributed_cc_grpc",
    srcs = [":distributed_proto"],
    grpc_only = True,
    deps = [
        ":distributed_cc_proto",
        "@grpc//:grpc++_reflection",
    ],
)
