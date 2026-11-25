# Copyright (c) 2026 Seoul National University
# SPDX-License-Identifier: MIT

package(
    default_visibility = ["//session:__subpackages__"],
)

cc_library(
    name = "network_util",
    srcs = [
        "network_util.cc",
    ] + select({
        "@platforms//os:android": ["android_network.cc"],
        "//conditions:default": [],
    }),
    hdrs = [
        "network_util.h",
    ] + select({
        "@platforms//os:android": ["android_network.h"],
        "//conditions:default": [],
    }),
    deps = [
        "//session/util:logger",
    ],
)
