# Copyright (c) 2026 Seoul National University
# SPDX-License-Identifier: MIT

import subprocess
import os


def validate_container_id(container_id):
    """Validate that the container ID exists and is running."""
    if not container_id:
        return True  # None is valid (will use default)

    try:
        result = subprocess.run(
            ['docker', 'ps', '--format', '{{.ID}}', '--filter', f'id={container_id}'],
            capture_output=True, text=True, check=True
        )
        if container_id in result.stdout:
            return True

        result = subprocess.run(
            ['docker', 'ps', '-a', '--format', '{{.ID}}', '--filter', f'id={container_id}'],
            capture_output=True, text=True, check=True
        )
        if container_id in result.stdout:
            print(f"Warning: Container {container_id} exists but is not running")
            return True

        print(f"Error: Container {container_id} not found")
        return False
    except subprocess.CalledProcessError as e:
        print(f"Error validating container ID: {e}")
        return False


def get_docker_util_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "docker_util.sh")


def get_container_hash(keyword="spatialblend-session"):
    if keyword in ["CONTAINER", "ID", "IMAGE", "COMMAND", "CREATED", "STATUS", "PORTS", "NAMES"]:
        return []
    return subprocess.getoutput(f"docker ps -a | grep {keyword} | awk '{{print $1}}'").split()


def copy_docker(src, dst, hash=None):
    if hash is None:
        hash = get_container_hash()[0]
    elif not validate_container_id(hash):
        raise ValueError(f"Invalid container ID: {hash}")

    os.makedirs(dst, exist_ok=True)

    import platform
    if platform.system() == "Windows":
        subprocess.call(['sh', get_docker_util_path(), '-d', hash, src, dst], shell=True)
    else:
        subprocess.call(['sh', get_docker_util_path(), '-d', hash, src, dst])


def run_cmd_docker(exec, hash=None):
    if hash is None:
        hash = get_container_hash()[0]
    elif not validate_container_id(hash):
        raise ValueError(f"Invalid container ID: {hash}")

    import platform
    if platform.system() == "Windows":
        subprocess.call(['sh', get_docker_util_path(), '-r', hash, exec], shell=True)
    else:
        subprocess.call(['sh', get_docker_util_path(), '-r', hash, exec])


def run_cmd_docker_with_output(exec, hash=None):
    """Run Docker command and capture output, returning lines as list."""
    if hash is None:
        hash = get_container_hash()[0]
    elif not validate_container_id(hash):
        raise ValueError(f"Invalid container ID: {hash}")

    import platform
    try:
        if platform.system() == "Windows":
            result = subprocess.run(['sh', get_docker_util_path(), '-r', hash, exec],
                                    shell=True, capture_output=True, text=True)
        else:
            result = subprocess.run(['sh', get_docker_util_path(), '-r', hash, exec],
                                    capture_output=True, text=True)

        output = result.stdout + result.stderr
        return output.splitlines() if output else []
    except Exception as e:
        return [f"Error running Docker command: {e}"]
