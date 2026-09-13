"""Read-only pinned training runtime fingerprint; imports but performs no inference."""
from pathlib import Path
import hashlib
import json
import platform
import sys
import numpy as np
import torch
import onnx
import onnxruntime as ort

def identity():
    paths=[Path(sys.executable).resolve(),Path(torch._C.__file__),Path(np.core._multiarray_umath.__file__)]
    paths+=list((Path(torch.__file__).parent/'lib').glob('libtorch_cpu.so'))
    paths+=list((Path(torch.__file__).parent/'lib').glob('*.dll'))
    paths+=list((Path(np.__file__).parent.parent/'numpy.libs').glob('*.so*'))
    paths+=list((Path(np.__file__).parent/'.libs').glob('*.dll'))
    paths+=list((Path(np.__file__).parent.parent/'numpy.libs').glob('*.dll'))
    paths+=list((Path(ort.__file__).parent/'capi').glob('*pybind11_state*.so'))
    paths+=list((Path(ort.__file__).parent/'capi').glob('*pybind11_state*.pyd'))
    paths+=list(Path(onnx.__file__).parent.glob('*cpp2py_export*.so'))
    paths+=list(Path(onnx.__file__).parent.glob('*cpp2py_export*.pyd'))
    paths+=list(Path(sys.executable).parent.glob('python310.dll'))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    return dict(python=platform.python_version(),python_build=sys.version,numpy=np.__version__,
        torch=torch.__version__,onnx=onnx.__version__,onnxruntime=ort.__version__,
        platform=platform.platform(),binary_sha256={str(p):sha(p) for p in sorted(set(paths))},
        inference_calls=0,optimizer_updates=0,physics_steps=0)

if __name__=='__main__':
    output=Path(__file__).resolve().parent.parent/'training_runtime_identity.json'
    assert not output.exists()
    result=identity();output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
