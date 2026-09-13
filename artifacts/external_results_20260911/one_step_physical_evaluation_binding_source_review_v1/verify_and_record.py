import ast,hashlib,json
from pathlib import Path
N=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')
B=N/'one_step_physical_student_evaluation_v1';O=Path(__file__).parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(name,value):
    with (O/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
frozen=read(B/'source_freeze.json');tests=read(B/'binding_workflow_tests.json');derivation=read(B/'binding_tools_derivation.json')
assert len(frozen['source_sha256'])==25
assert tests['passed'] is True and tests['tests']==7
assert tests['real_model_calls']==tests['native_steps']==tests['launchers_executed']==0
pins={}
for name,h in frozen['source_sha256'].items():
    p=B/'source_snapshot_v1'/name;assert sha(p)==h,name;pins[p.as_posix()]=h
    ast.parse(p.read_text(encoding='utf-8-sig'))
for name,h in tests['source_sha256'].items():
    p=B/name;assert sha(p)==h,name;pins[p.as_posix()]=h;ast.parse(p.read_text(encoding='utf-8-sig'))
assert derivation['new_sources']['freeze_final_package.py']=='85c836dff1d3cf451e20ac766b8e3133f3d27a7e506c31f54f98dc19a5b3e0e3'
assert derivation['new_sources']['prepare_bound_launcher.py']=='f2d0d4c3ae96aebde291954a3cf7300996f3e11edf529d348385a19d0a35848d'
for name,h in derivation['new_sources'].items():assert sha(B/name)==h
assert sha(B/'binding_workflow_tests.log')==tests['log_sha256']
prior=Path(frozen['review_path']);assert sha(prior)==frozen['review_sha256']=='3fef12f2357fadd76113ed5ee3eb256182e3fd162772ec75b32d933686366f73'
for name in ['source_freeze.json','source_preparation.json','source_only_check.json','freeze_sources.py','prepare_binding_tools.py','binding_tools_derivation.json','binding_workflow_tests.json','binding_workflow_tests.log','BINDING_WORKFLOW.md']:
    p=B/name;pins[p.as_posix()]=sha(p)
pins[prior.as_posix()]=sha(prior)
for p in (B/'launcher_template_checks').glob('*.ps1'):pins[p.as_posix()]=sha(p)
assert len(list((B/'launcher_template_checks').glob('*.ps1')))==4
for name in ['witness_binding.json','evaluation_binding.json','witness_process','evaluation_process','head_witness','nominal','post_lifecycle_hold_5s']:
    assert not (B/name).exists(),name
review=dict(kind='independent_physical_evaluation_binding_tools_source_review',passed=True,source_review_pass=True,
    preparation_review_pass=True,actual_binding_review_pass=False,witness_launch_authorized=False,canonical_launch_authorized=False,
    source_sha256=pins,direct_subject_sha256=pins,
    checks=['25 runtime files remain byte-identical to the prior prepared runtime review.',
    'Actual final 75000 head/checkpoint/fit report/training receipt and real physical dataset subjects are required before binding.',
    'Witness scope is exactly one batch-one WSL head call; evaluation scope is unchanged 1569 plus conditional continuous 250.',
    'Original raw combined-action history, strict native gates and actual first-activation witness parity remain unchanged.',
    'Mode-specific duplicate-output guards, execution/started CreateNew locks, hidden child native-handle capture, unknown/nonzero exit and pre/post source hashes reviewed.',
    'Seven owner pure/stub checks passed; generated templates are unbound and were not executed.'],findings=[],
    limits=['Only source preparation is cleared. Root stage selection and actual concrete binding/launcher review remain mandatory.'],
    reviewer_execution=dict(model_calls=0,native_steps=0,optimizer_updates=0,launches=0))
save('review.json',review)
print(json.dumps(dict(review_sha256=sha(O/'review.json'),source_pins=len(pins))))
