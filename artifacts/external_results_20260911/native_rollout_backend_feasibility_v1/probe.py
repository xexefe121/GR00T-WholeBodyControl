"""Inspect the installed native runtime's compiler and pointer interface; no steps."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import mujoco

out = Path(__file__).resolve().parent
package = Path(mujoco.__file__).resolve().parent
model_path = Path('/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/native_prepared.xml')
model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)
libraries = sorted(package.glob('libmujoco.so*'))
compiler = shutil.which('g++')
report = dict(kind='read_only_native_rollout_backend_runtime_probe',
              mujoco_python_version=mujoco.__version__, native_version=mujoco.mj_version(),
              package=str(package), header=str(package / 'include/mujoco/mujoco.h'),
              header_exists=(package / 'include/mujoco/mujoco.h').is_file(),
              model_pointer_interface=isinstance(getattr(model, '_address', None), int),
              data_pointer_interface=isinstance(getattr(data, '_address', None), int),
              native_dimensions=[model.nq, model.nv, model.nu],
              compiler=compiler,
              compiler_version=subprocess.check_output([compiler, '--version'], text=True).splitlines()[0] if compiler else None,
              libraries={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in libraries},
              inputs={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (model_path, Path(__file__))},
              physics_steps=0, compiled_code=False, controller_connected=False,
              purpose='Assess a possible native PD rollout backend after Python preview costs exceeded the 20 ms budget')
(out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report), flush=True)
