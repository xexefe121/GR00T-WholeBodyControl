"""Frozen-source and saved-receipt review only; no task imports or execution."""
import ast
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

BASE=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')
OUT=Path(__file__).resolve().parent
FIT=BASE/'direct_target_continuation_v1'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def subject(path):return {'path':Path(path).as_posix(),'sha256':sha(path)}

request_path=FIT/'training_request.json';receipt_path=FIT/'training_frozen_inputs.json'
assert sha(request_path)=='e3dbc8c41433699326cd611ea97767974b6eb9ce5de2c7b1f678f83594054625'
assert sha(receipt_path)=='1b9499a88b26b9feb5e1259d63284010e7521150c591a9815063cacc0251fa9c'
request=read(request_path);receipt=read(receipt_path)
pins=dict(receipt['input_sha256'])
sources={str(Path(receipt['source_directory'])/k):v for k,v in receipt['source_sha256'].items()}
pins.update(sources)
pins.update({str(request_path):sha(request_path),str(receipt_path):sha(receipt_path)})
for path,digest in pins.items():assert sha(path)==digest,path
assert len(receipt['input_sha256'])==189 and len(sources)==9
assert receipt['training_request_sha256']==sha(request_path)
assert request['root_selected'] is True and request['restored_start_step']==5000
assert request['updates']==50000 and request['ordinary_final_step']==55000
assert request['budgets']==dict(training_head_rows=705500000,diagnostic_torch_rows=460740,ORT_calls=601,native_steps=0,BFM_calls=0)
assert request['restoration']['checkpoint_sha256']=='ad6d657affba0796e7313d85ace240cd31d46e188c577da049880a7a031aecba'
original=BASE/'direct_target_student_v1/source_snapshot_v1'
unchanged=['direct_contract.py','direct_data.py','direct_model.py','direct_objective.py','direct_diagnostics.py','test_direct.py']
for name in unchanged:assert (original/name).read_bytes()==(FIT/'source_snapshot_v1'/name).read_bytes(),name
for path in sources:ast.parse(Path(path).read_text(encoding='utf-8-sig'))
tests=read(FIT/'tests/report.json');preflight=read(FIT/'saved_data_preflight.json')
assert tests['passed'] is True and tests['exit_code']==0 and tests['source_sha256']==receipt['source_sha256']
assert sha(FIT/'tests/stderr.log')==tests['stderr_sha256']
assert 'Ran 30 tests' in (FIT/'tests/stderr.log').read_text() and '\nOK\n' in (FIT/'tests/stderr.log').read_text()
assert preflight['passed'] is True and preflight['source_sha256']==receipt['source_sha256']
assert preflight['nominal_shape']==[9904,1000] and preflight['physical_shape']==[3054,1000]
assert preflight['sampler_shape']==[5000,576]
for rel in ['fit','training_clearance.json','fit_process/running.lock','fit_process/start.json','fit_process/child.json','fit_process/exit.json']:
    assert not (FIT/rel).exists(),'Prelaunch artifact already exists: '+rel
launcher=FIT/'run_fit_durable_v1.ps1'
assert sha(launcher)=='e9dd194b0bec222024187f90a0b533dd3c36bdf30f9916be4faa6731009f1cf7'
ps=(launcher).read_text()
for marker in ['[IO.FileMode]::CreateNew','[IO.FileShare]::Delete','-WindowStyle Hidden','$capturedHandle=$child.Handle','$child.WaitForExit();$rawExit=$child.ExitCode','if($null -eq $rawExit)','PYTHONPATH',"CUBLAS_WORKSPACE_CONFIG"]:assert marker in ps,marker
subjects={name:subject(path) for name,path in {
    'training_request':request_path,'frozen_inputs':receipt_path,'launcher':launcher,
    'trainer':FIT/'source_snapshot_v1/train_direct.py','continuation_contract':FIT/'source_snapshot_v1/continuation_contract.py',
    'source_checkpoint':Path(request['restoration']['checkpoint_path']),
    'prior_final_review':BASE/'direct_target_final_fit_review_v1/review.json',
    'prior_saved_audit':BASE/'direct_target_root_audit_v1/results_v1/report.json',
    'tests':FIT/'tests/report.json','saved_data_preflight':FIT/'saved_data_preflight.json',
    'methodology':BASE/'direct_target_continuation_methodology_v1/NOTE.md',
    'review_source':Path(__file__)}.items()}
review=dict(kind='direct_target_continuation_concrete_prelaunch_review',created_utc=datetime.now(timezone.utc).isoformat(),
    verdict='CLEAR',passed=True,source_review_pass=True,prelaunch_review_pass=True,
    training_request_sha256=sha(request_path),frozen_receipt_sha256=sha(receipt_path),subjects=subjects,
    source_sha256=receipt['source_sha256'],input_sha256=pins,verified_pin_count=len(pins),
    checks=dict(frozen_sources_and_inputs_exact=True,six_original_modules_byte_exact=True,synthetic_tests=30,
        saved_data_preflight_passed=True,task_outputs_and_execution_locks_absent=True,
        all_six_AdamW_states_RNG_and_normalization_restored_before_initial_pass=True,
        reused_initial_GPU_pass_compares_all_saved_predictions_and_metrics_before_updates=True,
        original_objective_and_batch_arithmetic_unchanged=True,
        saved_5000_schedule_repeated_ten_times_without_RNG_draws=True,
        inclusive_50000_cosine_3e5_to_3e6=True,
        ordinary_final_saved_before_export_without_checkpoint_selection=True,
        partial_call_optimizer_loss_and_returned_output_evidence_retained=True,
        hidden_single_launch_no_retry_and_known_exit_required=True,
        final_source_input_request_clearance_rehash_required=True),
    scope=dict(selected_runs=1,restored_start_step=5000,additional_updates=50000,ordinary_final_step=55000,
        training_rows=705500000,diagnostic_torch_rows=460740,ORT_calls=601,BFM_calls=0,native_steps=0,
        reviewer_task_model_calls=0,reviewer_optimizer_updates=0,reviewer_native_steps=0,
        synthetic_test_optimizer_fixtures_acknowledged=True),
    limitations=['Source and provenance clearance for the root-selected fixed continuation, not a completed-fit or behavioral pass.',
        'Loss tradeoffs and future closed-loop stability remain uncertain. Final numerical/export audit and separately bound witness/canonical are required.',
        'Synthetic fixture optimizer steps are not task-model fitting; no task model calls were made by this reviewer.'])
with (OUT/'review.json').open('x',encoding='utf-8') as f:json.dump(review,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'verdict':'CLEAR','review_sha256':sha(OUT/'review.json'),'pins':len(pins)}))
