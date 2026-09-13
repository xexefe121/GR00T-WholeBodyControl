"""Freeze one bounded feedback variant after pure tests; no runtime execution."""
from pathlib import Path
import ast
import difflib
import hashlib
import json
import subprocess
import sys
import numpy as np

BASE=Path(__file__).parent;DRAFT=BASE/'source_draft_v1';FROZEN=BASE/'source_snapshot_v1'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')

original=json.loads((BASE/'original_sources.json').read_text());original_directory=Path(original['original_directory'])
changed=[];patches=[]
for name,digest in original['source_sha256'].items():
    old=original_directory/name;new=DRAFT/name;assert sha(old)==digest,name
    if sha(new)!=digest:
        changed.append(name)
        patches.extend(difflib.unified_diff(old.read_text().splitlines(keepends=True),new.read_text().splitlines(keepends=True),fromfile='original/'+name,tofile='variant/'+name))
assert set(changed)=={'student_linear_runtime.py','proposal_evidence.py','evaluate_physical_response_student.py','evaluation_gate.py'}
old=(original_directory/'student_linear_runtime.py').read_text()
expected=old.replace('from terminal_yaw4_goal import terminal_goal_yaw4','from terminal_yaw4_goal import terminal_goal_yaw4\nfrom clipped_feedback import clipped_component_feedback')
expected=expected.replace('        self.seed.previous_action=combined.copy();self.seed.recorded_controls+=1',
    '        feedback,native_clip,feedback_clip=clipped_component_feedback(\n            combined,base+delta,target,self.c,learned=not terminal and not disable_head)\n        self.seed.previous_action=feedback.copy();self.seed.recorded_controls+=1')
expected=expected.replace('            state=sensed,history=history,features=x,inference_ms=(time.perf_counter()-tick)*1000)',
    '            raw_combined_action=combined.copy(),feedback_action=feedback.copy(),\n            native_target_clip_mask=native_clip,feedback_clip_mask=feedback_clip,\n            state=sensed,history=history,features=x,inference_ms=(time.perf_counter()-tick)*1000)')
assert expected==(DRAFT/'student_linear_runtime.py').read_text(),'Unexpected runtime edit'
old_ast=ast.parse((original_directory/'evaluate_physical_response_student.py').read_text())
new_ast=ast.parse((DRAFT/'evaluate_physical_response_student.py').read_text())
for name in ('get_state','assess','new_trace','source_metrics','zero_parity'):
    left=next(n for n in old_ast.body if getattr(n,'name',None)==name)
    right=next(n for n in new_ast.body if getattr(n,'name',None)==name)
    assert ast.dump(left)==ast.dump(right),name
for path in DRAFT.rglob('*.py'):ast.parse(path.read_text())
result=subprocess.run([sys.executable,str(BASE/'test_clipped_feedback.py')],capture_output=True,text=True)
with (BASE/'tests.log').open('x') as stream:stream.write(result.stdout+result.stderr)
assert result.returncode==0,result.stdout+result.stderr
write_new(BASE/'tests.json',dict(passed=True,test_count=9,exit_code=0,models_called=0,native_steps=0,optimizer_updates=0,
    test_source_sha256=sha(BASE/'test_clipped_feedback.py'),log_sha256=sha(BASE/'tests.log'),python=sys.version,numpy=np.__version__))
FROZEN.mkdir(exist_ok=False)
sources={}
for path in sorted(DRAFT.rglob('*.py')):
    relative=path.relative_to(DRAFT);target=FROZEN/relative;target.parent.mkdir(exist_ok=True,parents=True);target.write_bytes(path.read_bytes());sources[relative.as_posix()]=sha(target)
assert len(sources)==27
with (BASE/'source_derivation.patch').open('x') as stream:stream.write(''.join(patches))
sys.path.insert(0,str(FROZEN));from clipped_feedback import clipped_component_feedback
prior=BASE.parent/'one_step_physical_student_evaluation_v1/nominal/trace.npz'
contract=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json')
c=json.loads(contract.read_text())
with np.load(prior) as data:
    raw=data['action'][264];position=data['raw_proposal'][264];target=data['target'][264];actual=data['actual_normalized_action'][264]
    feedback,native,mask=clipped_component_feedback(raw,position,target,c,True)
    write_new(BASE/'clipped_control264_example.json',dict(control=264,raw_combined=raw.tolist(),raw_position=position.tolist(),
        applied_target=target.tolist(),applied_normalized=actual.tolist(),feedback_action=feedback.tolist(),
        native_target_clip_mask=native.tolist(),feedback_clip_mask=mask.tolist(),
        unclipped_components_bitexact=raw[~mask].tobytes()==feedback[~mask].tobytes(),
        original_trace_sha256=sha(prior),contract_sha256=sha(contract),no_models=True,no_physics=True))
write_new(BASE/'source_freeze.json',dict(kind='selected_clipped_component_feedback_source_freeze',source_directory=FROZEN.as_posix(),
    source_sha256=sources,original_source_directory=original_directory.as_posix(),original_source_sha256=original['source_sha256'],
    changed_originals=changed,unchanged_original_count=21,new_helpers=['clipped_feedback.py','feedback_parity.py'],
    only_runtime_edit_expected_text_exact=True,strict_physics_functions_AST_exact=True,
    original_trace_sha256=sha(prior),original_contract_sha256=sha(contract),
    selection_sha256=sha(BASE/'SELECTION.md'),tests_sha256=sha(BASE/'tests.json'),derivation_sha256=sha(BASE/'source_derivation.patch'),
    example_sha256=sha(BASE/'clipped_control264_example.json'),freeze_source_sha256=sha(__file__),
    source_only=True,actual_binding_created=False,models_called=0,native_steps=0))
print(json.dumps(dict(source_freeze_sha256=sha(BASE/'source_freeze.json'),sources=len(sources),tests=9,models_called=0,native_steps=0)))
