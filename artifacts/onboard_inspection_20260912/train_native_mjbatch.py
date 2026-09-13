"""Pin native MuJoCo before sharing the existing CUDA training dependencies."""
from pathlib import Path
import sys,runpy
import mujoco,numpy
if mujoco.__version__!='3.2.3':raise RuntimeError('Use the existing mjbatch323 interpreter')
ROOT=Path(__file__).resolve().parents[2]
for dependency in ('/mnt/e/codex_sonic_runtime/bfm_seed_20260910/onnx_deps',
    '/root/.venvs/g1_true23_mjlab/lib/python3.11/site-packages'):
    sys.path.append(dependency)
sys.path.insert(0,str(ROOT))
if '--native-physics' not in sys.argv:sys.argv.append('--native-physics')
runpy.run_module('gear_sonic.scripts.train_g1_true23_factory_dynamics',run_name='__main__')
