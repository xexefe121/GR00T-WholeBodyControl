import hashlib
import json
from pathlib import Path

N=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')
B=N/'one_step_physical_student_v1'
O=Path(__file__).parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(name,value):
    with (O/name).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def direct(value,digest):
    if isinstance(value,dict):return any(direct(v,digest) for v in value.values())
    if isinstance(value,list):return any(direct(v,digest) for v in value)
    return value==digest

subjects={
    (B/'selection_v1.json').as_posix():'63e46d3dc527b8c379bb4eb158257f4b62e1dc0b614477294c7b989c30ebd74a',
    (B/'training_request.json').as_posix():'bc0c4987fdfaf39b4a44b49bc44c7c84b0c0b24ed5fdc9ad9807a082c28aecce',
    (B/'training_frozen_inputs.json').as_posix():'ac61a9cfdcb32f4af210664ec20562e78bf3826f85bac4e7cf5e08f300167045',
    (B/'fit_launch_plan.json').as_posix():'646de4e412fc7580af75f7e0566ccf4a8ac3d7e12f934080dbf668421efee2f5',
    (N/'one_step_branch_combined_data_review_v1/review.json').as_posix():'6faf42e89281ac155fdc35a6b38f5b5218c2d067318c46b7a883cafc1797d9ed',
    (N/'one_step_physical_fit_source_review_v1/review.json').as_posix():'0e0f58a1c1269e9e3990127e63b8ccd21ba83f8d0d0d4cc15860fb9bae14f494'}
for p,h in subjects.items():assert sha(p)==h,p
selection=read(B/'selection_v1.json');request=read(B/'training_request.json')
frozen=read(B/'training_frozen_inputs.json');plan=read(B/'fit_launch_plan.json')
freeze=read(B/'freeze_report.json')
assert selection['selected_by']=='root' and selection['model_fitting_authorized'] is True
assert selection['automatic_optimizer_resume'] is False and selection['checkpoint_selection'] is False
assert selection['canonical_evaluation_selected'] is False
assert request['additional_updates']==frozen['additional_updates']==selection['additional_updates']==5000
assert plan['requested_updates']==5000
assert request['ordinary_final_step']==frozen['ordinary_final_step']==plan['ordinary_final_step']==selection['ordinary_final_step']==75000
assert request['valid_rows']==request['requested_rows']==selection['expected_valid_rows']==3054
assert request['nominal_rows']==3057
assert request['coefficients']==selection['coefficients']==dict(nominal=1.,velocity=1.,physical=1.)
assert request['requested_cell_denominators']==selection['requested_cell_denominators']==[99,819,100]*3
expected_budget=dict(valid_rows=3054,updates=5000,training_head_rows=36315000,legacy_head_onnx_calls=1126,
    physical_head_onnx_calls=24,head_onnx_calls=1150,diagnostic_torch_rows=293480,
    analytical_head_evaluations=14,BFM_calls=0,native_steps=0)
assert request['budgets']==frozen['budgets']==plan['budgets']==expected_budget
assert request['exact_conflict_groups']==0
assert request['compatibility']==dict(total_rows=6111,unique_feature_inputs=6111,duplicate_inputs=0,
    exact_target_conflicts=0,float32_residual_conflicts=0,signed_zero_canonicalized=True,rows_removed=0,targets_averaged=0)
for i,cell in enumerate(request['coverage']):
    assert cell['dataset']==i//3 and cell['phase']==('acquisition','source','return')[i%3]
    assert cell['requested']==cell['valid']==[99,819,100][i%3]
    assert cell['strict_failed']==0 and cell['effective_mass']==1/9 and cell['empty'] is False
assert len(request['coverage'])==9 and freeze['coverage']==request['coverage']
assert freeze['completed'] is True and freeze['valid_rows']==3054
assert freeze['model_evaluations']==freeze['optimizer_updates']==freeze['ORT_calls']==freeze['physics_steps']==0
snapshot=B/'source_snapshot_v1'
assert frozen['source_directory']==snapshot.as_posix()
assert len(frozen['source_sha256'])==36 and len(frozen['input_sha256'])==1518
assert set(p.relative_to(snapshot).as_posix() for p in snapshot.rglob('*') if p.is_file())==set(frozen['source_sha256'])
pins={}
def add(p,h):
    key=Path(p).as_posix()
    if key in pins:assert pins[key]==h,key
    pins[key]=h
for obj in (selection,request,frozen,plan):
    for p,h in obj['input_sha256'].items():add(p,h)
for name,h in frozen['source_sha256'].items():
    add(snapshot/name,h)
    assert sha(B/'source_draft_v2'/name)==h,name
for role,value in request['reviews'].items():
    add(value['path'],value['sha256'])
    review=read(value['path']);assert review[value['pass_field']] is True,role
    for p,h in value['subjects'].items():
        assert direct(review,h),(role,p)
        assert frozen['input_sha256'].get(p)==h or request['reviews']['source']['subjects'].get(p)==h,(role,p)
        add(p,h)
assert len(request['reviews']['source']['subjects'])==39
assert len(request['reviews']['branch_data']['subjects'])==9
assert frozen['training_request_sha256']==plan['training_request_sha256']==sha(B/'training_request.json')
assert plan['frozen_receipt_sha256']==sha(B/'training_frozen_inputs.json')
assert plan['selection_sha256']==request['selection_sha256']==frozen['selection_sha256']==sha(B/'selection_v1.json')
assert plan['automatic_resume'] is False
command=dict(python='C:/Users/camer/AppData/Local/Programs/Python/Python310/python.exe',
    driver=(snapshot/'fit_physical_continuation.py').as_posix(),working_directory=snapshot.as_posix(),
    arguments=['-u',(snapshot/'fit_physical_continuation.py').as_posix()],
    environment=dict(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=snapshot.as_posix()))
assert plan['command']==command
for p,h in subjects.items():add(p,h)
add(B/'freeze_report.json',sha(B/'freeze_report.json'))
for p,h in pins.items():assert sha(p)==h,p
absent=['training_clearance.json','fit_launch_receipt.json','fit','fit_process','fit_preflight_failure.json','freeze_failure.json','finalize_failure.json']
for name in absent:assert not (B/name).exists(),name
verification=dict(passed=True,input_sha256=pins,distinct_files_verified=len(pins),frozen_input_count=1518,
    snapshot_sources=36,review_roles=4,source_subjects=39,data_subjects=9,command=command,budgets=expected_budget,
    actual_artifacts_absent=absent,reviewer_model_calls=0,reviewer_optimizer_updates=0,reviewer_native_steps=0)
save('verification.json',verification)
review=dict(kind='independent_selected_physical_fit_prelaunch_review',passed=True,source_review_pass=True,
    final_frozen_request_review_pass=True,model_fitting_authorized=True,final_launch_review_pass=True,
    direct_subject_sha256=subjects,source_sha256=request['reviews']['source']['subjects'],
    branch_data_subject_sha256=request['reviews']['branch_data']['subjects'],
    prior_fit_export_subject_sha256=request['reviews']['prior_fit']['subjects'],
    verification=dict(path=(O/'verification.json').as_posix(),sha256=sha(O/'verification.json')),
    command=command,budgets=expected_budget,additional_updates=5000,ordinary_final_step=75000,
    fixed_valid_rows=3054,nominal_rows=3057,requested_cell_denominators=[99,819,100]*3,
    verdict='CLEAR for the root-selected single hidden fit after the reviewed finalizer binds this receipt. No additional root permission required.',
    findings=[],automatic_resume=False,checkpoint_selection=False,canonical_launch_authorized=False,
    limitations=['Fit/export completion and independent saved-evidence review remain required before any selected controller evaluation. This review makes no behavioral success claim.'],
    reviewer_execution=dict(model_calls=0,optimizer_updates=0,native_steps=0))
save('review.json',review)
print(json.dumps(dict(review_path=(O/'review.json').as_posix(),review_sha256=sha(O/'review.json'),
    verification_sha256=sha(O/'verification.json'),files_verified=len(pins))))
