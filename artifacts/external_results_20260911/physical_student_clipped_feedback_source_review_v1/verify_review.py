"""Read-only frozen source review; no runtime/model/native imports."""
import ast
import hashlib
import json
from pathlib import Path

BASE=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')
OWNER=BASE/'physical_student_clipped_feedback_v1'
OUT=Path(__file__).parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
pins={}
def pin(p,expected=None):
    p=Path(p);d=sha(p)
    if expected is not None:assert d==expected,(str(p),d,expected)
    pins[p.as_posix()]=d
    return d
f=read(OWNER/'source_freeze.json')
pin(OWNER/'source_freeze.json','d4096deacebe360022729fafa9b2aa64f4d9d60b753ebad571b13624f327055a')
source=Path(f['source_directory']);original=Path(f['original_source_directory'])
for n,d in f['source_sha256'].items():
    pin(source/n,d)
    assert sha(OWNER/'source_draft_v1'/n)==d
for n,d in f['original_source_sha256'].items():pin(original/n,d)
changed=[n for n,d in f['original_source_sha256'].items() if f['source_sha256'][n]!=d]
assert sorted(changed)==sorted(f['changed_originals']) and len(changed)==4
assert len(f['source_sha256'])==27 and len(f['original_source_sha256'])-len(changed)==21
def functions(p):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
old=functions(original/'evaluate_physical_response_student.py')
new=functions(source/'evaluate_physical_response_student.py')
for n in old:
    if n!='main':assert old[n]==new[n],n
tests=read(OWNER/'tests.json');assert tests['passed'] is True and tests['test_count']==9 and tests['exit_code']==0
for key in ('models_called','native_steps','optimizer_updates'):assert tests[key]==0
pin(OWNER/'tests.json',f['tests_sha256'])
pin(OWNER/'test_clipped_feedback.py',tests['test_source_sha256']);pin(OWNER/'tests.log',tests['log_sha256'])
for name,key in [('source_derivation.patch','derivation_sha256'),('clipped_control264_example.json','example_sha256'),('freeze_sources.py','freeze_source_sha256'),('SELECTION.md','selection_sha256')]:pin(OWNER/name,f[key])
for name in ('freeze_bound_evaluation.py','prepare_bound_launcher.py'):
    ast.parse((OWNER/name).read_text());pin(OWNER/name)
for p in [BASE/'student_physical_response_saved_outcome_v1/report.json',BASE/'student_physical_response_fixed_map_v1/report.json',BASE/'student_physical_response_fixed_map_v1/completion_review.json',BASE/'one_step_physical_student_evaluation_v1/nominal/trace.npz']:
    pin(p)
assert pins[(BASE/'one_step_physical_student_evaluation_v1/nominal/trace.npz').as_posix()]==f['original_trace_sha256']
pin(Path(__file__))
report=dict(kind='selected_clipped_component_feedback_source_review',source_review_pass=True,source_review_passed=True,
    source_sha256=f['source_sha256'],input_sha256=pins,
    changed_originals=changed,unchanged_originals=21,new_helpers=2,
    evidence=dict(owner_tests_passed=9,all_snapshot_and_draft_hashes_exact=True,original_top_level_nonmain_evaluator_functions_AST_exact=True,
        runtime_change='Only learned native-clipped components use applied-target inverse. Every unmasked float32 component, raw command log, BFM startup and terminal feedback retain the original arithmetic.',
        prefix_gate='All controls 0..264 and 2650 native samples, full291 precontrol265 state, history, warnings and repeated clock compared before the first changed input; comparison only.',
        timing_gate='First current prior changes265; actor lag remains exact265 and changes only newest action23 block at266. No speculative history or state injection.',
        accounting='Four new per-command fields appended by existing generic recording loop. Empty/partial trace shape gates and raw/returned failure evidence preserved.',
        binding_helpers='Existing75000 head and exact one-call witness reused; CLI permits evaluation only, requested1569 plus conditional250, CreateNew locks, hidden child handle/exit and source/input pre/post hashes.'),
    limitations=['Source review is not launch clearance; actual immutable binding and generated launchers still require final review.','Controlled late-amplification test only: initial departures251 and260 precede clipping264.','Matching fixed-map ankle target at319 did not imply physical safety; fixed maps are not a replanned expert or feasibility proof.'],
    reviewer_model_calls=0,reviewer_native_steps=0,reviewer_optimizer_updates=0,hardware_authorized=False)
for p,d in pins.items():assert sha(p)==d,p
with (OUT/'review.json').open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2);stream.write('\n')
print(json.dumps(dict(source_review_pass=True,review_sha256=sha(OUT/'review.json'),pins=len(pins))))
