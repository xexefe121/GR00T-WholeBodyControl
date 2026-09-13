"""Concrete source/input launch review; hashes and saved metadata only."""
import ast
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone

BASE=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911/direct_target_full_state_student_v1')
OUT=Path(__file__).resolve().parent
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
pins={}
def checked(path,digest):
    path=Path(path);assert sha(path)==digest,str(path);pins[path.as_posix()]=digest
    return path
request_path=checked(BASE/'training_request.json','0c3bbc4e8567f6dfc09dfe7d9cdfcdd8af5a56178ea112af3a647d84c48d0dea')
frozen_path=checked(BASE/'training_frozen_inputs.json','bbb67b48187dcead0dc0886496573447ffdd58582eaeb63f41a7fa947e922f18')
launcher=checked(BASE/'run_fit_durable_v1.ps1','5ff6a81edde3a9ab131dbce99714fd39a7ed875c8fa7b8554fce822905173e41')
request=read(request_path);frozen=read(frozen_path)
assert frozen['training_request_sha256']==sha(request_path)
assert frozen['source_file_count']==len(frozen['source_sha256'])==15
assert frozen['input_count']==len(frozen['input_sha256'])==216
for path,digest in frozen['input_sha256'].items():checked(path,digest)
for name,digest in frozen['source_sha256'].items():
    path=checked(Path(frozen['source_directory'])/name,digest)
    ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
assert Path(frozen['source_directory']).resolve()==(BASE/'source_snapshot_v1').resolve()
assert request['kind']=='one_full_state_finite_feedback_fit' and request['root_selected'] is True
assert request['updates']==request['optimizer_step']==10000 and request['ordinary_final_step']==65000
assert request['sampler_seed']==20260911 and request['features']==1000 and request['head_output']=='normalized_target'
assert not request['automatic_retry'] and not request['checkpoint_selection'] and not request['canonical_evaluation_authorized'] and not request['hardware_authorized']
proposal=read(BASE/'proposal.json')
for field in ('objective','sampler','optimizer','export'):assert request[field]==proposal[field]
assert request['budgets']==dict(calibration_forward_rows=14686,calibration_forward_calls=3,calibration_gradient_calls=3,
    training_forward_rows=146860000,training_forward_calls=30000,diagnostic_Torch_rows=1470280,diagnostic_Torch_calls=5748,
    diagnostic_ORT_rows=367570,diagnostic_ORT_calls=1437,native_calls=0,BFM_calls=0,manual_export_trace_calls=0)
for name,subject in request['subjects'].items():
    assert pins[Path(subject['path']).as_posix()]==subject['sha256']
    if 'pass_field' in subject:
        actual=read(subject['path']);assert actual[subject['pass_field']] is True
        for key,value in subject['required_fields'].items():assert actual[key]==value
assert request['subjects']['checkpoint']['sha256']=='9ceef5099ebd154e08a1c2c3784c4e06021d464e607474548763665b24f8f9e7'
assert request['subjects']['full_state_root_owner']['sha256']=='3f47a899d3e08ebb9d5a1ec514e395563dd677c4209a281f774f76401b401e43'
assert request['subjects']['full_state_data_review']['sha256']=='22505dccc1b68e984be6811e08b6774b7e22b72c6af391e3e6ae5affa4062568'
for field in ('paths','full_state_paths','restoration_predictions'):
    for path in request[field].values():assert Path(path).as_posix() in pins
tests=read(BASE/'tests/report.json');assert tests['passed'] is True and tests['exit_code']==0 and tests['synthetic_only'] is True
assert tests['source_sha256']==frozen['source_sha256']
assert 'Ran 24 tests' in (BASE/'tests/stderr.log').read_text(encoding='utf-8')
preflight=read(BASE/'saved_data_preflight.json');assert preflight['passed'] is True
assert preflight['source_sha256']==frozen['source_sha256']
for path,digest in preflight['input_sha256'].items():assert pins[Path(path).as_posix()]==digest
assert preflight['nominal_shape']==[9904,1000] and preflight['full_state_shape']==[354612,1000] and preflight['physical_shape']==[3054,1000]
assert preflight['full_state_center_cells']==[100,819,100]*3 and preflight['physical_cells']==[99,819,100]*3
assert preflight['group_axes']==[3,3,23,3,3,23]
runtime=request['runtime'];assert read(runtime['verification_path'])[runtime['pass_field']] is True
assert pins[Path(runtime['identity_path']).as_posix()]==runtime['identity_sha256']
for path,digest in read(runtime['identity_path'])['source_binary_sha256'].items():assert pins[Path(path).as_posix()]==digest
driver=(BASE/'source_snapshot_v1/train_full_state.py').read_text(encoding='utf-8')
assert 'from restoration_support import exact_saved,restore_rng' in driver and 'verify_start(' not in driver
assert "optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5,foreach=False,fused=False)" in driver
assert "calibration_forward_outputs.npz" in driver and "forward_outputs_sha256=sha(dest/'calibration_forward_outputs.npz')" in driver
assert "active.clear();active.update(stage=kind" in driver
assert frozen['owner_checker']['path'].endswith('/verify_completed_v2.py')
assert pins[frozen['owner_checker']['path']]==frozen['owner_checker']['sha256']
assert not (BASE/'fit').exists() and not (BASE/'training_clearance.json').exists()
assert not (BASE/'fit_process/running.lock').exists() and not (BASE/'fit_process/start.json').exists()
subjects={name:dict(path=path.as_posix(),sha256=sha(path)) for name,path in (
    ('training_request',request_path),('frozen_inputs',frozen_path),('launcher',launcher),
    ('data_review',Path(request['subjects']['full_state_data_review']['path'])),
    ('root_data_audit',Path(request['subjects']['full_state_root_audit']['path'])),
    ('root_data_owner',Path(request['subjects']['full_state_root_owner']['path'])))}
result=dict(passed=True,prelaunch_review_pass=True,source_review_pass=True,created_utc=datetime.now(timezone.utc).isoformat(),
    training_request_sha256=sha(request_path),frozen_receipt_sha256=sha(frozen_path),subjects=subjects,
    input_sha256=pins,source_sha256=frozen['source_sha256'],all_current_pins_exact=True,frozen_input_count=216,source_count=15,
    fixed_scope=dict(start_global_step=55000,ordinary_final_step=65000,fresh_optimizer=True,optimizer_updates=10000,
        nominal_rows=9904,full_state_rows=354612,physical_rows=3054,full_state_cells=54,pairs_per_update=864,
        calibration='one saved initial N/F/P gradient measurement; fixed norm(N)/norm(F)',budgets=request['budgets']),
    findings=[],resolved_findings=['Successful calibration now retains actual three forward arrays and cell vectors, hash-bound before gradients; no extra calls.',
        'Original restoration helper only supplies exact_saved and restore_rng; old5000 optimizer restoration is not called.',
        'Active return evidence cleared per calibration/update, each current graph flag distinguishes returned/synchronized/verified.'],
    verification=['Read source math, fixed sampling and loss arithmetic, exact restoration/fresh optimizer, release gates, and full failure preservation.',
        'Read concrete freezer and durable hidden launcher; literal request/review/source/runtime pins, .NET shared-delete hashing, CreateNew attempt and captured handle/known-exit guards.',
        'All216 input pins and15 source files rehashed;24 frozen synthetic tests and saved full-data schema/mapping preflight passed.'],
    no_fit_outputs_or_attempt_present=True,task_model_calls=0,calibration_calls=0,optimizer_updates=0,native_steps=0,
    actual_fit_selected_by_root=True,one_hidden_fit_launch_cleared=True,canonical_evaluation_cleared=False,hardware_authorized=False,
    limitations=['Source and data qualification do not establish learned closed-loop stability; ordinary-final saved evidence and numerical export still require independent review.'],
    review_source_sha256=sha(__file__))
with (OUT/'review.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
print((OUT/'review.json').as_posix(),sha(OUT/'review.json'))
