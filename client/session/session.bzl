# Copyright (c) 2026 Seoul National University
# SPDX-License-Identifier: MIT

def session_test_cxxopts():
    return ["-DSPATIALBLEND_TEST"]

def session_linkopts():
    return select({
        "@platforms//os:android": [
            "-lc++_static",  # Static C++ runtime
            "-lc++abi",  # C++ ABI support library
            "-llog",  # Android logging
            "-lm",  # libm
            "-lc",  # libc
            "-latomic",  # Required for some uses of ISO C++11 <atomic> in x86.
            "-ldl",  # Required for `dl_iterate_phdr`
            "-Wl,--no-export-dynamic",
            "-Wl,--gc-sections",
            "-Wl,--as-needed",
        ],
        "//conditions:default": [],
    })

def session_shared_lib_name():
    libname = select({
        "//session:debug": "libspatialblend_session_d",
        "//session:release": "libspatialblend_session",
        "//conditions:default": "libspatialblend_session",
    })
    extension = select({
        "@platforms//os:linux": ".so",
        "@platforms//os:android": ".so",
        "@platforms//os:macos": ".dylib",
        "@platforms//os:windows": ".dll",
        "//conditions:default": ".so",
    })
    return libname + extension

def session_export_inputs():
    """Linker inputs that limit the shared library to the spatialblend_* C API."""
    return select({
        "@platforms//os:linux": ["//session/c:version_script.lds"],
        "@platforms//os:android": ["//session/c:version_script.lds"],
        "@platforms//os:macos": ["//session/c:exported_symbols.txt"],
        "//conditions:default": [],
    })

def session_export_linkopts():
    return select({
        "@platforms//os:linux": ["-Wl,--version-script=$(location //session/c:version_script.lds)"],
        "@platforms//os:android": ["-Wl,--version-script=$(location //session/c:version_script.lds)"],
        "@platforms//os:macos": [
            "-Wl,-exported_symbols_list,$(location //session/c:exported_symbols.txt)",
            "-Wl,-install_name,@rpath/libspatialblend_session.dylib",
        ],
        "//conditions:default": [],
    })

def session_cc_test(name, linkopts = [], **kwargs):
    native.cc_test(
        name = name,
        copts = session_test_cxxopts(),
        linkopts = linkopts + session_linkopts(),
        **kwargs
    )

def session_version():
    return "0.0.1"
