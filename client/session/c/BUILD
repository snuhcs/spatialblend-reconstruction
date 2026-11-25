# Copyright (c) 2026 Seoul National University
# SPDX-License-Identifier: MIT

load("@rules_pkg//pkg:tar.bzl", "pkg_tar")
load("//session:session.bzl", "session_cc_test", "session_export_inputs", "session_export_linkopts", "session_linkopts", "session_shared_lib_name")

cc_library(
    name = "session_c",
    srcs = ["spatialblend_session.cc"],
    hdrs = [
        "spatialblend_session.h",
        "spatialblend_session_type.h",
        "spatialblend_session_utility.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//session/inference:inference_client",
        "//session/inference/distributed:sam3_seg_client",
        "//session/inference/distributed:sam3d_obj_client",
        "//session/inference/sam3d:sam3d_client",
        "//session/util:logger",
        "//session/util:status_ring",
        "//session/util:util",
        "@abseil-cpp//absl/strings",
    ],
    alwayslink = True,
)

cc_shared_library(
    name = "c_shared",
    additional_linker_inputs = session_export_inputs(),
    shared_lib_name = session_shared_lib_name(),
    user_link_flags = session_linkopts() + session_export_linkopts(),
    deps = [
        "//session/c:session_c",
    ],
)

filegroup(
    name = "c_api_headers",
    srcs = glob(["**/*.h"]),
)

# PDB files for debug builds (Windows only)
filegroup(
    name = "c_shared_pdb",
    srcs = [":c_shared"],
    output_group = "pdb_file",
)

pkg_tar(
    name = "libspatialblend_session_tar",
    srcs = [":c_shared"],
    package_dir = "lib",
)

pkg_tar(
    name = "headers_tar",
    srcs = [":c_api_headers"],
    package_dir = "include/spatialblend_session",
)

pkg_tar(
    name = "libspatialblend_session_c",
    deps = [
        ":headers_tar",
        ":libspatialblend_session_tar",
    ],
    srcs = select({
        "//session:debug": [":c_shared_pdb"],
        "//conditions:default": [],
    }),
)

session_cc_test(
    name = "c_api_test",
    srcs = ["c_api_test.cc"],
    deps = [
        ":session_c",
    ],
)
