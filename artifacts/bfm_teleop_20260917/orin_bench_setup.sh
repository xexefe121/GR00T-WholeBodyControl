#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/unitree/bfm_orin_bench
if [ -e "$ROOT" ]; then
  echo "refusing: $ROOT already exists" >&2
  exit 73
fi
mkdir -p "$ROOT"/{bin,python,cache,src/gear_sonic/utils,data,weights,reports}
export UV_INSTALL_DIR="$ROOT/bin"
export UV_PYTHON_INSTALL_DIR="$ROOT/python"
export UV_CACHE_DIR="$ROOT/cache"
export UV_NO_MODIFY_PATH=1
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$ROOT/bin:$PATH"
uv python install 3.10
uv venv --python 3.10 "$ROOT/venv"
uv pip install --python "$ROOT/venv/bin/python" torch==2.4.1 mujoco==3.2.3 safetensors numpy==1.26.4 scipy pyyaml pyzmq
"$ROOT/venv/bin/python" - <<'PY'
import importlib.metadata as m, json, platform, sys
names = ('torch', 'mujoco', 'safetensors', 'numpy', 'scipy', 'PyYAML', 'pyzmq')
versions = {name: m.version(name) for name in names}
versions.update(python=sys.version, machine=platform.machine())
open('/home/unitree/bfm_orin_bench/reports/installed_versions.json', 'w').write(json.dumps(versions, indent=2) + '\n')
PY
