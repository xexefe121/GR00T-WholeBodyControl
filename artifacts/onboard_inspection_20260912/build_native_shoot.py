"""Build small native3.2.3 lookahead library; no robot or scheduled job."""
from pathlib import Path
import argparse
import os
import subprocess
import mujoco
ROOT=Path(__file__).resolve().parents[2]
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
mode=ap.add_mutually_exclusive_group();mode.add_argument('--feedback',action='store_true');mode.add_argument('--response',action='store_true');mode.add_argument('--preview',action='store_true');a=ap.parse_args()
assert mujoco.__version__=='3.2.3'
a.output.mkdir(parents=True,exist_ok=True);package=Path(mujoco.__file__).parent
name='libtrue23preview.so' if a.preview else 'libtrue23response.so' if a.response else 'libtrue23shoot.so'
temporary=a.output/(name+'.building')
source='true23_preview.cpp' if a.preview else 'true23_response.cpp' if a.response else 'true23_feedback_shoot.cpp' if a.feedback else 'true23_shoot.cpp'
subprocess.run(['g++',*(['-O2','-ffp-contract=off','-fopenmp'] if a.preview else ['-O3']),'-std=c++17','-shared','-fPIC',*(['-fopenmp','-march=native'] if a.feedback or a.response else []),'-I'+str(package/'include'),
    str(ROOT/'gear_sonic/native'/source),str(next(package.glob('libmujoco.so*'))),
    '-Wl,-rpath,'+str(package),'-o',str(temporary)],check=True)
os.replace(temporary,a.output/name);print(a.output/name)
