# Copyright (c) 2026 Seoul National University
# SPDX-License-Identifier: MIT

package(
    default_visibility = ["//session:__subpackages__"],
)

cc_library(
    name = "logger",
    srcs = ["logger.cc"],
    hdrs = ["logger.h"],
    deps = [
        "//session/common",
        "@abseil-cpp//absl/strings",
    ],
)

cc_library(
    name = "status_ring",
    srcs = ["status_ring.cc"],
    hdrs = ["status_ring.h"],
    deps = [
        "@abseil-cpp//absl/status",
    ],
)

cc_library(
    name = "util",
    hdrs = ["util.h"],
    deps = [
        "//session/common",
    ],
)
