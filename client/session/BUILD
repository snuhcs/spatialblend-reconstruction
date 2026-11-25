# Copyright (c) 2026 Seoul National University
# SPDX-License-Identifier: MIT

# Root BUILD file for session package

config_setting(
    name = "debug",
    values = {
        "compilation_mode": "dbg",
    },
)

config_setting(
    name = "release",
    values = {
        "compilation_mode": "opt",
    },
)

platform(
    name = "android_arm64",
    constraint_values = [
        "@platforms//cpu:arm64",
        "@platforms//os:android",
    ],
)
