"""Read-only Linux invocation of actual inherited frozen() via isolated AST.

Only the pure function is compiled. No task module, NumPy, MuJoCo, ORT or model
is imported. This verifies the producer's actual conversion and hashing logic.
"""
import ast
import hashlib
import json
import sys
from pathlib import Path

base=Path(__file__).resolve().parent
source=base/'source_snapshot_v1/run_actual_student_oracle.py'
tree=ast.parse(source.read_text())
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='frozen')
checked={}
def sha256(path):
    digest=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    checked[str(path)]=digest
    return digest
env=dict(BASE=base,__file__=str(source),Path=Path,json=json,sha256=sha256)
exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),env)
receipt=env['frozen']()
assert len(checked)==len(receipt['source_sha256'])+len(receipt['input_sha256'])
assert all('\\' not in name for name in receipt['input_sha256'])
request=json.loads((base/'execution_request.json').read_text())
for role in request['subjects'].values():
    assert receipt['input_sha256'][role['path']]==role['sha256']
    name=role['path']
    if len(name)>2 and name[1]==':':name='/mnt/'+name[0].lower()+name[2:]
    assert checked[name]==role['sha256']
result=dict(passed=True,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    request_sha256=hashlib.sha256((base/'execution_request.json').read_bytes()).hexdigest(),
    frozen_receipt_sha256=hashlib.sha256((base/'frozen_inputs.json').read_bytes()).hexdigest(),
    actual_inherited_frozen_function_ast=True,checked_sha256=checked,
    checked_sources=len(receipt['source_sha256']),checked_inputs=len(receipt['input_sha256']),
    task_modules_imported=False,model_calls=0,native_calls=0,replans=0)
assert not any(x in sys.modules for x in ('numpy','mujoco','onnxruntime','torch'))
print(json.dumps(result,indent=2))
