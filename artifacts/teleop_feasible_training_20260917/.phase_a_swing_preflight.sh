#!/usr/bin/env bash
set -euo pipefail
source /root/venvs/teleop23/bin/activate
cd /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof
python - <<'PY'
import importlib.util
from pathlib import Path

script=Path('gear_sonic/scripts/prepare_g1_true23_twist2_replay.py')
spec=importlib.util.spec_from_file_location('twist_importer',script)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
path=Path('/mnt/z/codex/twist2_inspect/assets/ref_motions/accad_A3___Swing_t2.pkl')
try:
    value=module.NumpyOnlyUnpickler(path.open('rb')).load()
    print({'type':type(value).__name__,'keys':sorted(value) if isinstance(value,dict) else None,
           'shapes':{key:getattr(value[key],'shape',None) for key in value} if isinstance(value,dict) else None})
except Exception as exc:
    print({'error':f'{type(exc).__name__}: {exc}'})
PY
