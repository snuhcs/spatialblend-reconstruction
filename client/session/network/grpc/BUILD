# Copyright (c) 2026 Seoul National University
# SPDX-License-Identifier: MIT

package(
    default_visibility = ["//session:__subpackages__"],
)

cc_library(
    name = "grpc",
    hdrs = ["grpc_client_base.h"],
    copts = [
        "-Wno-unused-parameter",
        "-Wunknown-warning-option",
        "-Wattributes",
        "-Wunknown-attributes",
    ] + select({
        "@platforms//os:windows": ["/std:c++17"],
        "//conditions:default": ["-std=c++17"],
    }),
    deps = [
        "//session/network/util:network_util",
        "//session/util:logger",
        "//session/util:util",
        "@grpc//:grpc++_reflection",
    ],
)
