# Copyright (c) 2026 Seoul National University
# SPDX-License-Identifier: MIT

import builtins
import os
import argparse
import platform as plf
import shlex
import shutil
import subprocess

# ANSI color codes
BLUE = '\033[94m'
RESET = '\033[0m'
PREFIX = f"{BLUE}SPATIALBLEND-SESSION:{RESET} "

# Save the original print function
original_print = builtins.print


def custom_print(*args, **kwargs):
    """Custom print function that adds the SpatialBlend Session prefix to all prints."""
    original_print(PREFIX, *args, **kwargs)


# Override the default print function
builtins.print = custom_print

PLATFORM = {
    "linux": "linux",
    "darwin": "macos",
    "macos": "macos",
    "android": "android",
    "windows": "windows"
}


def is_windows():
    return plf.system() == "Windows"


def is_linux():
    return plf.system() == "Linux"


def is_macos():
    return plf.system() == "Darwin"


def is_cygwin():
    return plf.system().startswith("CYGWIN_NT")


def get_platform():
    return plf.system().lower()


def canon_path(path):
    if is_windows() or is_cygwin():
        return path.replace('\\', '/')
    return path


def get_output_base_path(docker: bool, debug: bool, platform: str):
    if docker:
        home_dir = "/root"
    elif is_windows():
        home_dir = "C:/"
    else:
        home_dir = os.path.expanduser("~")
    build_type = "d" if debug else "o"

    if platform == "host":
        platform_name = PLATFORM[get_platform()]
    else:
        platform_name = PLATFORM[platform]

    if is_windows():
        # Short path to avoid Windows MAX_PATH (260 char) limit.
        # Deeply nested external deps (e.g. envoy_api) exceed 260 chars
        # with longer output_base paths.
        short = {"windows": "w", "linux": "l", "macos": "m", "android": "a"}
        p = short.get(platform_name, platform_name[0])
        return canon_path(os.path.join(home_dir, "_b", f"s_{p}_{build_type}"))

    return canon_path(os.path.join(home_dir, ".cache", "bazel",
                                   f"sb_session_{platform_name}_{build_type}"))


def get_bazel_options(
        debug: bool,
        platform: str,
        strip_never: bool = False):
    opt = "--copt=-DSPATIALBLEND_BUILD_DLL "
    opt += "-c " + ("dbg" if debug else "opt") + " "
    opt += "--strip " + ("never" if (debug or strip_never) else "always") + " "
    if platform != "host" and platform != "linux":
        opt += f"--config {PLATFORM[platform]} "
    return opt


def _maybe_fix_python_executable(args):
    if not args:
        return args
    if args[0] != "python":
        return args
    if shutil.which("python"):
        return args
    if shutil.which("python3"):
        args[0] = "python3"
    return args


def run_cmd(cmd):
    args = shlex.split(cmd)
    args = _maybe_fix_python_executable(args)
    subprocess.call(args, cwd=os.getcwd())


def run_cmd_with_output(cmd):
    """Run command and capture output, returning lines as list."""
    args = shlex.split(cmd)
    args = _maybe_fix_python_executable(args)
    try:
        result = subprocess.run(args, cwd=os.getcwd(), capture_output=True, text=True)
        output = result.stdout + result.stderr
        return output.splitlines() if output else []
    except Exception as e:
        return [f"Error running command: {e}"]


def copy(src, dst, overwrite=False):
    if not os.path.exists(src):
        print(f"Error: Source file/directory does not exist: {src}")
        return False

    if not os.path.exists(dst):
        os.makedirs(os.path.normpath(dst), exist_ok=True)

    dst = canon_path(os.path.join(dst, os.path.basename(src)))

    try:
        if os.path.isdir(src):
            shutil.copytree(src, dst, ignore_dangling_symlinks=True,
                            symlinks=True, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        return True
    except FileNotFoundError:
        print(f"Error: Could not find source file: {src}")
        return False
    except Exception as e:
        print(f"Error copying {src} to {dst}: {str(e)}")
        return False


def make_cmd(
    docker: bool,
    build_only: bool,
    debug: bool,
    platform: str,
    target: str,
    strip_never: bool = False,
):
    cmd = "bazel" + " "
    cmd += "--output_base " + \
        get_output_base_path(docker, debug, platform) + " "
    cmd += ("build" if build_only else "test") + " "
    cmd += get_bazel_options(debug, platform, strip_never)
    cmd += target
    return cmd


def get_dst_path(base_dir, target_platform, debug):
    build = 'debug' if debug else 'release'
    path = os.path.join(base_dir, target_platform, build)
    return canon_path(path)


def get_argument_parser(desc: str):
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-all', action="store_true", default=False,
                        help='Build all platforms (default = false)')
    parser.add_argument('-a', '--android', action="store_true", default=False,
                        help='Build android (default = false)')
    parser.add_argument('-d', '--debug', action="store_true", default=False,
                        help='Build debug (default = release)')
    parser.add_argument('-r', '--rebuild', action="store_true", default=False,
                        help='Re-build test target, only affects to android')
    parser.add_argument('-b', '--build', action="store_true", default=True,
                        help='Build only, only affects to local (default = true)')
    parser.add_argument('-docker', action="store_true", default=False,
                        help='Compile / Pull cross-compiled binaries from docker')
    parser.add_argument('-dh', '--docker_hash', type=str, default=None,
                        help='Docker container ID/hash to use for building. If not specified, will auto-detect the first spatialblend-session container.')
    parser.add_argument('-sn', '--strip-never', action="store_true", default=False,
                        help='Force Bazel --strip=never even for release builds')
    parser.add_argument('--verbose', action="store_true", default=True,
                        help='Show build output in terminal (default: quiet, output saved to files)')
    return parser
