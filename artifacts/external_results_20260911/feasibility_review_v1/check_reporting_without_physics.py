"""Execute only JSON conversion and empty-shape export AST; never run controller."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np

path=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/gear_sonic/scripts/evaluate_g1_true23_mjbatch_mpc.py')
source_bytes=path.read_bytes()
tree=ast.parse(source_bytes)
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='finite_json')
scope={'np':np}
exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),scope)
original={'failure':{'kind':'hard_physical_feasibility','time':float('nan')},
          'metrics':{'peak':np.float64('inf'),'nested':[1.,float('-inf')]}}
converted=scope['finite_json'](original)
assert converted=={'failure':{'kind':'hard_physical_feasibility','time':None},'metrics':{'peak':None,'nested':[1.,None]}}
json.dumps(converted,allow_nan=False)
assert np.isnan(original['failure']['time']) and np.isposinf(original['metrics']['peak'])
run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
shape_block=next(n for n in run.body if isinstance(n,ast.If) and any(
    isinstance(c,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='trailing_shapes' for t in c.targets)
    for c in n.body))
keys=['target','planned_target','planned_state','feedback_gain','joint_error','physics_torque',
      'physics_requested_torque','physics_actuator_force','fresh_seed_measured_history']
arrays={name:np.asarray([]) for name in keys}
scope.update(hard_feasibility=True,arrays=arrays,TRACKED=tuple(range(6)))
exec(compile(ast.Module(body=[shape_block],type_ignores=[]),str(path),'exec'),scope)
expected=dict(target=(0,23),planned_target=(0,23),planned_state=(0,59),feedback_gain=(0,23,58),
              joint_error=(0,23),physics_torque=(0,23),physics_requested_torque=(0,23),
              physics_actuator_force=(0,23),fresh_seed_measured_history=(0,300))
for key,shape in expected.items():assert arrays[key].shape==shape
report=dict(scope='Pure reporting operations only; no controller, MjData or physics',
            source_sha256=hashlib.sha256(source_bytes).hexdigest(),
            json_nonfinite_summaries_to_null=True,original_nonfinite_inputs_unmodified=True,
            empty_export_shapes={k:list(v.shape) for k,v in arrays.items()})
assert source_bytes==path.read_bytes(), 'source changed during pure reporting witness'
Path(__file__).with_name('reporting_runner_snapshot.py').write_bytes(source_bytes)
Path(__file__).with_name('reporting_checks.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
