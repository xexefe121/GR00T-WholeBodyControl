"""Run only after parent confirms no competing timing benchmark/canonical."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import contextlib,json,sys
from pathlib import Path
import torch
import pytest
import onnx

BASE=Path(__file__).resolve().parent;SOURCE=BASE/'source_draft_v1'
torch.set_num_threads(1);torch.set_num_interop_threads(1)
assert not torch.cuda.is_initialized()
sys.path.insert(0,str(SOURCE))
with (BASE/'runtime.json').open('x') as f:
    json.dump(dict(python=sys.version,executable=sys.executable,torch_version=torch.__version__,onnx_version=onnx.__version__,
        torch_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),cuda_initialized=False),f,indent=2)
with (BASE/'synthetic_tests.log').open('x') as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
    code=pytest.main([str(SOURCE/'test_width512_promoted.py'),'--rootdir',str(BASE),'--confcutdir',str(BASE),'-p','no:cacheprovider','-q',
        '--basetemp='+str(BASE/'synthetic_only_temp'),'--junitxml='+str(BASE/'synthetic_tests.xml')])
assert not torch.cuda.is_initialized()
with (BASE/'test_exit.json').open('x') as f:
    json.dump(dict(exit_code=int(code),cuda_initialized=False,actual_checkpoint_loads=0,task_arrays_loaded=0,
        Torch_forward_calls=0,ORT_sessions=0,ORT_calls=0,native_steps=0,optimizer_updates=0),f,indent=2)
print(json.dumps(dict(exit_code=int(code),cuda_initialized=False,Torch_forward_calls=0,ORT_calls=0)))
raise SystemExit(code)
