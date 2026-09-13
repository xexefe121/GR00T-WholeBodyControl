import hashlib,json
from pathlib import Path
N=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911');B=N/'one_step_physical_student_v1'
A=N/'physical_fit_evidence_independent_v1';O=Path(__file__).parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(name,value):
    with (O/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
subjects={
    (B/'fit/report.json').as_posix():'5153ef3e193059062cf071b2b82c1961f7465683e8d8c9e5471dd89a27b25a1b',
    (B/'fit/student_head.pt').as_posix():'9f31d74855c28a57231c51e0a652a5f2eeeaac87a3c5aff032d59ba5ca981e34',
    (B/'fit/student_head.onnx').as_posix():'fb856003734acc0338586482968a7e31553a11826e549a8b486e0662e4934e81',
    (A/'report.json').as_posix():'d52a859e78f4c1bb6e84ef24cc110fcb548777ff58e854fc2a88d51071f8685a',
    (A/'request.json').as_posix():'334b07e749dbeecb041fe053f1a38d6864dd42311e7f58b5fffd7f87ef72e11c',
    (B/'training_frozen_inputs.json').as_posix():'ac61a9cfdcb32f4af210664ec20562e78bf3826f85bac4e7cf5e08f300167045',
    (B/'training_request.json').as_posix():'bc0c4987fdfaf39b4a44b49bc44c7c84b0c0b24ed5fdc9ad9807a082c28aecce',
    (B/'fit_launch_receipt.json').as_posix():'1245984f798beb2ae2ad2c55fe86983fc051219448e2c98149ab36219fce7135',
    (N/'one_step_physical_fit_prelaunch_review_v1/review.json').as_posix():'ae96ce17b9bfaaaaf802fede9361cdbd2adeb3c0530cc2bfbe3f910fea859748',
    (N/'one_step_branch_combined_data_review_v1/review.json').as_posix():'6faf42e89281ac155fdc35a6b38f5b5218c2d067318c46b7a883cafc1797d9ed',
    (N/'physical_fit_independent_audit_source_review_v1/review.json').as_posix():'0c829aa1b6a3be8fe627d7fc3d22635a14ddb2c96071bfab67dff69c64df7156'}
for p,h in subjects.items():assert sha(p)==h,p
report=read(B/'fit/report.json');audit=read(A/'report.json');request=read(A/'request.json')
owner=read(B/'completion_verification_v1.json');launch=read(B/'fit_launch_receipt.json')
frozen=read(B/'training_frozen_inputs.json');training=read(B/'training_request.json')
assert report['ordinary_final_step']==audit['ordinary_final_optimizer_steps']==owner['completed_steps']==75000
assert report['additional_updates']==owner['additional_updates']==5000
for key in ('completed','optimization_completed','final_export_diagnostics_completed','numerical_gate_passed','export_parity_passed'):
    assert report[key] is True,key
assert report['checkpoint_selection'] is False and report['physical_evaluations']==report['new_expert_queries']==0
assert report['hardware_authorized'] is False and audit['passed'] is True and owner['passed'] is True
assert report['head_ONNX_calls']==owner['head_ONNX_calls']==1150
assert report['final_head_parity']==owner['final_export_parity']==2.7418136596679688e-6
assert report['final_head_parity']<1e-5
assert report['checkpoint_sha256']==subjects[(B/'fit/student_head.pt').as_posix()]
assert report['onnx_sha256']==subjects[(B/'fit/student_head.onnx').as_posix()]
assert audit['source_sha256']=='1aace720c1c25eef6b1706d135fbef686858e4250bfc11a96c8a38035928245f'
assert audit['helper_sha256']=='615026c216f0635bb7af6ba7d08a0e0005563424e0582f0255a8b773174cf7b9'
assert sha(A/'audit_physical_fit_evidence.py')==audit['source_sha256']
assert sha(A/'audit_physical_fit_math.py')==audit['helper_sha256']
assert audit['request_sha256']==sha(A/'request.json')
for key in ('all2880000_private_sampler_pairs_and_final_Torch_RNG_exact','NumPy_RNG_unchanged',
    'all5000_learning_rates_three_loss_compositions_and_cell_reductions_exact','original_norm_span_exact',
    'ONNX_weights_and_graph_exact','all3054_physical_identities_validity_and_requested_cell_weights_exact',
    'initial_all_legacy_predictions_byteexact_to70000','initial_final_all_prediction_metrics_exact',
    'fixed_first24_requested_windows_and_clip_replan_zero_gain_groups_exact','all_inputs_and_sources_unchanged'):
    assert audit[key] is True,key
assert audit['model_evaluations']==audit['ORT_calls']==audit['optimizer_updates']==audit['physics_steps']==0
assert audit['requested_physical_rows']==audit['valid_physical_rows']==3054 and audit['strict_failed_rows']==0
assert audit['budgets']==training['budgets'] and audit['coverage']==training['coverage']==report['physical_coverage']
assert request['private_generator_calls']==45000 and request['private_generator_row_axis_pairs']==2880000
assert request['required_training_receipt_sha256']==sha(B/'training_frozen_inputs.json')
count=0
with (A/'comparisons.jsonl').open(encoding='utf-8') as f:
    for count,line in enumerate(f,1):
        row=json.loads(line);assert row['number']==count and row['passed'] is True,count
assert count==audit['comparisons']==191851
status=read(B/'fit/attempt_status.json');exit_result=read(B/'fit_process/exit.json')
assert status['stage']=='COMPLETE'
assert exit_result['exit_code']==exit_result['raw_python_exit_code']==0 and exit_result['error'] is None
assert exit_result['exit_known'] is True and exit_result['all_postrun_pins_exact'] is True
assert exit_result['automatic_resume'] is False
assert all(p['absent'] is True and p['running'] is False for p in owner['process_states'])
for counter in (status,report['attempt_counters']):
    assert counter['completed_steps']==75000 and counter['optimizer_step_counters']==[75000]*6
    assert counter['committed_sample_rows']==counter['committed_loss_rows']==5000
    for key,num in [('training_head_rows',36315000),('head_onnx_calls',1150),('diagnostic_torch_rows',293480),('analytical_head_evaluations',14)]:
        assert counter[key+'_attempted']==counter[key+'_returned']==num,key
pins=dict(subjects)
def add(p,h):
    p=Path(p).as_posix()
    if p in pins:assert pins[p]==h,p
    pins[p]=h
for obj in (request,launch,frozen):
    for p,h in obj['input_sha256'].items():add(p,h)
for p,c in owner['input_checks'].items():
    assert c['matched'] is True and c['actual']==c['expected'];add(p,c['actual'])
for name,h in owner['output_sha256'].items():add(B/name,h)
for name,h in frozen['source_sha256'].items():add(B/'source_snapshot_v1'/name,h)
for p in (B/'completion_verification_v1.json',A/'comparisons.jsonl',A/'audit_physical_fit_evidence.py',A/'audit_physical_fit_math.py'):
    add(p,sha(p));subjects[p.as_posix()]=sha(p)
for p,h in pins.items():assert sha(p)==h,p
nominal={stage:read(B/f'fit/{stage}_nominal_metrics.json') for stage in ('initial','final')}
metrics={}
for name in ('nominal','velocity','physical_response','physical_absolute'):
    initial=audit['summaries']['initial'][name];final=audit['summaries']['final'][name]
    metrics[name]=dict(initial=initial,final=final,percent_change=(final/initial-1)*100)
metrics['nominal_applied_target_rmse_rad']={stage:nominal[stage]['all']['applied_target_rmse_rad'] for stage in nominal}
assert metrics['nominal']['percent_change']>119 and metrics['nominal_applied_target_rmse_rad']['final']>metrics['nominal_applied_target_rmse_rad']['initial']
save('verification.json',dict(passed=True,files_verified=len(pins),input_sha256=pins,root_comparisons_checked=count,
    metrics=metrics,model_calls=0,native_steps=0,optimizer_updates=0))
review=dict(kind='independent_ordinary75000_fit_export_review',passed=True,fit_review_passed=True,export_review_passed=True,
    fit_review_pass=True,export_review_pass=True,ordinary_final_step=75000,additional_updates=5000,
    direct_subject_sha256=subjects,subjects={p:dict(path=p,sha256=h) for p,h in subjects.items()},
    verification=dict(path=(O/'verification.json').as_posix(),sha256=sha(O/'verification.json')),
    metrics=metrics,export_max_delta_rad=report['final_head_parity'],head_ONNX_calls=1150,
    training_head_rows=36315000,diagnostic_torch_rows=293480,analytical_head_evaluations=14,
    restored_70000_model_AdamW_RNG_and_norm_exact=True,all_six_optimizer_counters_75000=True,
    root_independent_saved_evidence_passed=True,checkpoint_selection=False,retraining_selected=False,
    behavioral_qualification=False,hardware_authorized=False,canonical_launch_authorized=False,
    verdict='CLEAR final numerical fit/export evidence for the root-selected witness and later canonical evaluation, subject to each concrete binding/launcher review.',
    qualifications=['Nominal objective worsened by 119.58 percent and nominal applied-target RMSE increased from about 0.06112 to 0.08090 rad. This is retained explicitly.',
        'Physical response, velocity response and physical absolute diagnostic losses improved; these are saved-data results, not feedback stability evidence.',
        'The fixed ordinary-final-only experiment requires finite numerical outputs and export parity; nominal improvement was not an added selection gate.',
        'A separately counted WSL batch-one activation witness and the unchanged canonical lifecycle remain required. No new model, optimizer or physics calls were made for this review.'],
    findings=[])
save('review.json',review)
print(json.dumps(dict(review_sha256=sha(O/'review.json'),verification_sha256=sha(O/'verification.json'),files_verified=len(pins),metrics=metrics)))
