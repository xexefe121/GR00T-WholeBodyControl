"""Create the concrete selected pair package, without training or approval."""
import hashlib
import json
from pathlib import Path
import shutil

BASE=Path(__file__).resolve().parent
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def item(path,**extra):return dict(path=Path(path).as_posix(),sha256=sha(path),**extra)

def launcher():
    old=BASE.parent/'direct_target_full_state_student_v1/run_fit_durable_v1.ps1'
    text=old.read_text(encoding='utf-8')
    substitutions=[
      ("$clear.additional_updates -ne 10000 -or $clear.ordinary_final_step -ne 65000", "$clear.updates_per_condition -ne 3000 -or $clear.ordinary_final_step -ne 68000 -or (@($clear.conditions) -join ',') -ne 'blinded,causal'"),
      ("$request.kind -ne 'one_full_state_finite_feedback_fit' -or $request.root_selected -ne $true -or $request.updates -ne 10000 -or $request.sampler_seed -ne 20260911", "$request.kind -ne 'matched_causal_context_utility_study' -or $request.root_selected -ne $true -or $request.updates_per_condition -ne 3000 -or (@($request.conditions) -join ',') -ne 'blinded,causal' -or $request.coefficient -ne 1.8188207859141674"),
      ("'train_full_state.py'", "'train_context_pair.py'"),
      ("kind='one_full_state_finite_feedback_fit'", "kind='matched_causal_context_utility_study'"),
      ("updates=10000;ordinary_final_step=65000", "updates_per_condition=3000;ordinary_final_step=68000;conditions=@('blinded','causal')"),
      ("kind='one_full_state_finite_feedback_fit_exit'", "kind='matched_causal_context_utility_study_exit'"),
    ]
    for before,after in substitutions:
        if text.count(before)!=1:raise ValueError('Launcher substitution identity: '+before)
        text=text.replace(before,after)
    begin=text.index("    $report=Read-SharedJson (Join-Path $runRoot 'fit/report.json')")
    end=text.index('    $runExit=0',begin)
    text=text[:begin]+'''    $pair=Read-SharedJson (Join-Path $runRoot 'fit/paired_report.json')
    if($pair.completed -ne $true -or $pair.updates_per_condition -ne 3000 -or $pair.ordinary_final_step -ne 68000 -or (@($pair.conditions) -join ',') -ne 'blinded,causal'){throw 'Fixed pair incomplete.'}
    foreach($field in @('shared_initial_actor_optimizer_RNG_normalization_exact','shared_schedule_exact','all_frozen_inputs_unchanged')){if($pair.$field -ne $true){throw "Pair proof failed: $field"}}
    if($pair.budgets.training_forward_rows -ne 88116000 -or $pair.budgets.training_forward_calls -ne 18000 -or $pair.budgets.training_updates -ne 6000){throw 'Pair budget differs.'}
    foreach($condition in @('blinded','causal')){
        $conditionRoot=Join-Path (Join-Path $runRoot 'fit') $condition
        $reportPath=Join-Path $conditionRoot 'report.json';$report=Read-SharedJson $reportPath
        if((Read-SharedSha $reportPath) -ne $pair.condition_report_sha256.$condition){throw 'Condition report identity differs.'}
        foreach($field in @('completed','optimization_completed','final_export_diagnostics_completed','numerical_gate_passed','export_parity_passed','all_frozen_inputs_unchanged')){if($report.$field -ne $true){throw "Incomplete $condition report: $field"}}
        if($report.condition -ne $condition -or $report.ordinary_final_step -ne 68000 -or $report.additional_updates -ne 3000 -or $report.optimizer_step -ne 3000 -or $report.features -ne 1323 -or $report.head_output -ne 'normalized_target'){throw 'Final condition metadata differs.'}
        if($report.counts.training.rows_verified -ne 44058000 -or $report.counts.training.calls_verified -ne 9000 -or $report.counts.calibration_forward_calls -ne 0 -or $report.counts.calibration_gradient_calls -ne 0){throw 'Condition training counts differ.'}
        foreach($backend in @('initial_GPU32','final_GPU32','CPU64','GPU64','ORT64')){if($report.counts.diagnostics.$backend.rows_verified -ne 367570 -or $report.counts.diagnostics.$backend.calls_verified -ne 1437){throw "Incomplete backend counts: $condition/$backend"}}
        if((Read-SharedSha (Join-Path $conditionRoot 'student_head.pt')) -ne $report.checkpoint_sha256 -or (Read-SharedSha (Join-Path $conditionRoot 'student_head.onnx')) -ne $report.onnx_sha256){throw 'Output identities differ.'}
    }
'''+text[end:]
    target=BASE/'run_fit_durable_v1.ps1'
    with target.open('x',encoding='utf-8',newline='\n') as f:f.write(text)
    write(BASE/'launcher_derivation.json',dict(original=item(old),result=item(target),substitutions=substitutions,
        replaced_success_verdict='Exact completed pair plus both condition reports, model subjects and fixed counts; raw child exit separately preserved',task_calls=0))
    monitor=(BASE.parent/'direct_target_full_state_student_v1/read_fit_progress.ps1').read_text(encoding='utf-8')
    monitor=monitor.replace("'fit/progress.json','fit/failure.json'", "'fit/blinded/progress.json','fit/causal/progress.json','fit/blinded/failure.json','fit/causal/failure.json','fit/paired_failure.json','fit/paired_report.json'")
    with (BASE/'read_fit_progress.ps1').open('x',encoding='utf-8',newline='\n') as f:f.write(monitor)

def freeze():
    prep=read(BASE/'source_preparation.json');proof=read(BASE/'context_preflight/report.json')
    review_path=BASE.parent/'direct_target_context_source_review_v1/review.json';review=read(review_path)
    if prep['source_preparation_passed'] is not True or proof['passed'] is not True or review['source_review_pass'] is not True or review['context_data_review_pass'] is not True:raise ValueError('Source/proof review required.')
    if review['source_sha256']!=prep['source_sha256']:raise ValueError('Reviewed source map differs.')
    pins=dict(read(BASE/'preparation_inputs.json')['input_sha256'])
    request=read(BASE/'training_request_proposal.json')
    request.update(root_selected=True,source_preparation_selected=True,actual_execution_clearance_absent=True,
        pending_before_actual_fit=['root concrete request/launcher clearance'],automatic_retry=False,
        source_preparation_sha256=sha(BASE/'source_preparation.json'),context_proof_sha256=sha(BASE/'context_preflight/report.json'))
    request['subjects'].update(
        context_source_review=item(review_path,pass_field='source_review_pass',required_fields=dict(context_data_review_pass=True)),
        context_preparation=item(BASE/'source_preparation.json',pass_field='source_preparation_passed'),
        context_proof=item(BASE/'context_preflight/report.json',pass_field='passed'))
    for role in request['subjects'].values():pins[role['path']]=role['sha256']
    for subject in prep['subjects'].values():pins[subject['path']]=subject['sha256']
    for path in sorted((BASE/'context_preflight').iterdir()):
        if path.is_file():pins[path.as_posix()]=sha(path)
    for name in ('prepare_execution.py','run_fit_durable_v1.ps1','read_fit_progress.ps1','launcher_derivation.json','verify_completed.py','test_execution_helpers.py','execution_helper_tests.xml','launcher_ast.json'):
        path=BASE/name;pins[path.as_posix()]=sha(path)
    for path,digest in pins.items():
        if sha(path)!=digest:raise ValueError('Changed pin: '+path)
    target=BASE/'source_snapshot_v1';target.mkdir(exist_ok=False)
    for name,digest in prep['source_sha256'].items():
        original=Path(prep['source_directory'])/name
        if sha(original)!=digest:raise ValueError('Changed prepared source: '+name)
        shutil.copyfile(original,target/name)
        if sha(target/name)!=digest:raise ValueError('Source copy differs: '+name)
    write(BASE/'training_request.json',request)
    receipt=dict(source_directory=target.as_posix(),source_sha256=prep['source_sha256'],input_sha256=pins,
        training_request_sha256=sha(BASE/'training_request.json'),all_inputs_rehashed=True,
        source_review_sha256=sha(review_path),context_proof_sha256=sha(BASE/'context_preflight/report.json'),
        task_model_calls=0,optimizer_updates=0,native_steps=0)
    write(BASE/'training_frozen_inputs.json',receipt)
    write(BASE/'training_clearance_draft.json',dict(approved=False,request_sha256=sha(BASE/'training_request.json'),
        frozen_receipt_sha256=sha(BASE/'training_frozen_inputs.json'),launcher_path=(BASE/'run_fit_durable_v1.ps1').as_posix(),
        launcher_sha256=sha(BASE/'run_fit_durable_v1.ps1'),conditions=['blinded','causal'],updates_per_condition=3000,
        ordinary_final_step=68000,automatic_retry=False,review_pass_field='prelaunch_review_pass'))
    print(json.dumps(dict(request_sha256=sha(BASE/'training_request.json'),frozen_receipt_sha256=sha(BASE/'training_frozen_inputs.json'),
        launcher_sha256=sha(BASE/'run_fit_durable_v1.ps1'),inputs=len(pins),sources=len(prep['source_sha256']))))

if __name__=='__main__':
    import sys
    if sys.argv[1:]==['launcher']:launcher()
    elif sys.argv[1:]==['freeze']:freeze()
    else:raise SystemExit('Use launcher or freeze; neither launches task models.')
