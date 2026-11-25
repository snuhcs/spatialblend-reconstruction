#!/usr/bin/env bash
# Copyright (c) 2026 Seoul National University
# SPDX-License-Identifier: MIT

# The scan server from a fresh clone: values -> Python environment -> SAM 3D Objects -> weights -> (optionally) run.
#
#   server/scripts/setup_server.sh                 # GPU host (Linux x86-64, NVIDIA, CUDA 12.1 toolkit): full install + weights
#   server/scripts/setup_server.sh --run           # ... then start the server (foreground)
#   server/scripts/setup_server.sh --cpu           # any OS (Windows: Git Bash), no GPU: CPU environment, tests and the stub (no models)
#
# Options:
#   --cpu             install only the CPU part (pip install -e '.[test]'), run the tests and a stub health probe
#   --skip-models     do not download the weights (they are already there; HF_TOKEN is then not required)
#   --run             start sam3d-scan-server at the end (GPU mode)
#   --venv DIR        the Python environment to create or reuse (default server/.venv); --venv active uses the
#                     environment that is already active (for example a conda env made from environment.yml)
#
# Values (scripts/configure_secrets.py): the environment wins over server/.env. HF_TOKEN is the only secret. The script
# takes it (or HUGGING_FACE_HUB_TOKEN) out of its environment at the start and keeps it in an unexported variable, so pip,
# install_sam3d_objects.sh, the tests, the stub and the server never see it; it is passed only to configure_secrets.py
# (to check it) and to the one sam3d-scan-models download call, and never printed. SAM3D_OBJECTS_ROOT defaults
# to ~/sam-3d-objects. Prerequisites: git, Python 3.11 (or uv, which fetches it), and on a GPU host a CUDA 12.1
# toolkit (CUDA_HOME; a conda cuda-nvcc 12.1 install works) for the source-built extensions.
set -euo pipefail

server=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
cpu=0 skip_models=0 run=0 venv="$server/.venv"
while (($#)); do
  case "$1" in
    --cpu) cpu=1 ;;
    --skip-models) skip_models=1 ;;
    --run) run=1 ;;
    --venv) venv=${2:?--venv needs a directory or "active"}; shift ;;
    -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done

# 1. Values. The token leaves the environment first, before any program runs, and lives in unexported variables;
# with_token hands it to one command.
hf_token=${HF_TOKEN:-} hub_token=${HUGGING_FACE_HUB_TOKEN:-}
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN

step() { printf '\n== %s\n' "$*"; }
# Windows (Git Bash): "python3" may be the Microsoft Store stub, and a venv keeps its programs in Scripts/, not bin/.
python3=python3
python3 -c '' 2>/dev/null || python3=python
bin=bin
[[ $(uname -s) == MINGW* || $(uname -s) == MSYS* ]] && bin=Scripts

with_token() {
  if [[ -n $hf_token ]]; then HF_TOKEN=$hf_token "$@"
  elif [[ -n $hub_token ]]; then HUGGING_FACE_HUB_TOKEN=$hub_token "$@"
  else "$@"; fi
}
# server/.env is read as KEY=VALUE lines (never executed); a value already in the environment wins.
step "values (server/.env and the environment)"
token_args=()
((cpu || skip_models)) && token_args=(--no-token)
if [[ -t 0 && $cpu == 0 ]]; then
  with_token "$python3" "$server/scripts/configure_secrets.py" ${token_args[@]+"${token_args[@]}"}
else
  with_token "$python3" "$server/scripts/configure_secrets.py" --check ${token_args[@]+"${token_args[@]}"}
fi
if [[ -f "$server/.env" ]]; then
  while IFS= read -r line || [[ -n $line ]]; do
    line=${line%$'\r'}
    [[ $line =~ ^[[:space:]]*(export[[:space:]]+)?(HF_TOKEN|SAM3D_[A-Z0-9_]+|HF_HOME)[[:space:]]*=[[:space:]]*(.*)$ ]] || continue
    key=${BASH_REMATCH[2]} value=${BASH_REMATCH[3]}
    value=${value%"${value##*[![:space:]]}"}   # trailing whitespace, as configure_secrets.py strips it
    if [[ ${#value} -ge 2 && ( ( $value == \"*\" ) || ( $value == \'*\' ) ) ]]; then value=${value:1:${#value}-2}; fi
    [[ -n $value ]] || continue
    if [[ $key == HF_TOKEN ]]; then
      [[ -n $hf_token ]] || hf_token=$value      # unexported: see with_token
    elif [[ -z ${!key:-} ]]; then
      export "$key=$value"
    fi
  done < "$server/.env"
fi
export SAM3D_OBJECTS_ROOT=${SAM3D_OBJECTS_ROOT:-$HOME/sam-3d-objects}

# 2. Python environment.
step "Python environment"
if [[ $venv == active ]]; then
  [[ -n ${VIRTUAL_ENV:-}${CONDA_PREFIX:-} ]] || { echo "--venv active: no virtualenv or conda env is active" >&2; exit 2; }
else
  if [[ ! -x "$venv/$bin/python" ]]; then
    if command -v uv >/dev/null; then
      uv venv --python 3.11 --seed "$venv"
    else
      py=$(command -v python3.11 || true)
      [[ -n $py ]] || { echo "need Python 3.11 (or uv, which installs it): https://docs.astral.sh/uv/" >&2; exit 2; }
      "$py" -m venv --upgrade-deps "$venv"
    fi
  fi
  # shellcheck disable=SC1091
  source "$venv/$bin/activate"
fi
python -m pip install --quiet --upgrade pip
cd "$server"

if ((cpu)); then
  step "CPU install, tests and stub"
  python -m pip install --quiet -e '.[test]'
  python -m pytest -q
  port=${SAM3D_SCAN_PORT:-9800}
  sam3d-scan-server --stub --host 127.0.0.1 --port "$port" &
  stub=$!
  trap 'kill $stub 2>/dev/null || true' EXIT
  sam3d-scan-server health --server "127.0.0.1:$port" --wait 60
  python tools/probe_stub.py "127.0.0.1:$port"
  echo "CPU setup OK: tests pass and the stub answers on 127.0.0.1:$port. A GPU host runs this script without --cpu."
  exit 0
fi

[[ $(uname -s) == Linux ]] || { echo "the GPU server needs Linux x86-64 with an NVIDIA GPU; use --cpu here" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: no NVIDIA driver on this host" >&2; exit 2; }

# 3. The GPU stack (environment.yml's pip part) and SAM 3D Objects at the pinned commit.
step "GPU dependencies"
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1
python -m pip install -e '.[gpu]'
step "SAM 3D Objects -> $SAM3D_OBJECTS_ROOT"
./scripts/install_sam3d_objects.sh "$SAM3D_OBJECTS_ROOT"

# 4. Weights (gated; the token goes to this one process only).
if ((skip_models)); then
  sam3d-scan-models --sam3d-root "$SAM3D_OBJECTS_ROOT" --check
else
  step "model weights"
  with_token sam3d-scan-models --sam3d-root "$SAM3D_OBJECTS_ROOT"
fi
unset hf_token hub_token HF_TOKEN HUGGING_FACE_HUB_TOKEN

# 5. Run.
if ((run)); then
  step "server (SAM3D_SCAN_HOST=${SAM3D_SCAN_HOST:-127.0.0.1} SAM3D_SCAN_PORT=${SAM3D_SCAN_PORT:-9800})"
  exec sam3d-scan-server --sam3d-root "$SAM3D_OBJECTS_ROOT"
fi
echo "Setup done. Start: $venv/bin/sam3d-scan-server --sam3d-root $SAM3D_OBJECTS_ROOT (or this script with --run)"
