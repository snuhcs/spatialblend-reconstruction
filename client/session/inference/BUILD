# Copyright (c) 2026 Seoul National University
# SPDX-License-Identifier: MIT

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "inference_client",
    srcs = ["inference_client.cc"],
    hdrs = ["inference_client.h"],
    deps = [
        "//session/network/grpc",
        "//session/network/util:network_util",
        "//session/util:logger",
        "//session/util:util",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@grpc//:grpc++",
    ],
)
