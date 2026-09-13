"""Build the experiment's native clock loop against the pinned MuJoCo3.2.3."""
from pathlib import Path
import subprocess
import os
import mujoco
import argparse

root=Path('/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
ap=argparse.ArgumentParser()
ap.add_argument('--output',type=Path,default=Path('/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/native_clock_v1'))
out=ap.parse_args().output
out.mkdir(parents=True,exist_ok=True)
package=Path(mujoco.__file__).parent
assert mujoco.__version__=='3.2.3'
library=next(package.glob('libmujoco.so*'))
temporary=out/'libtrue23clock.building.so'
subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC','-I'+str(package/'include'),
    str(root/'gear_sonic/native/true23_clock.cpp'),str(library),'-Wl,-rpath,'+str(package),
    '-o',str(temporary)],check=True)
# Replace the file only after a successful build; existing loaded mappings stay
# valid when another simulation already has the previous shared object open.
os.replace(temporary,out/'libtrue23clock.so')
print(out/'libtrue23clock.so')
