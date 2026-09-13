"""Bind completed training evidence without qualifying the failed FP32 export."""
import hashlib,json
from pathlib import Path

BASE=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')
OUT=Path(__file__).parent
FIT=BASE/'direct_target_continuation_v1'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p,expected=None):
    actual=sha(p)
    if expected is not None:assert actual==expected,(str(p),actual,expected)
    return {'path':str(p),'sha256':actual}

subjects={
 'fit_report':pin(FIT/'fit/report.json','dc8d834e8b7ef193cb6212248e4c35ca2cd7dab3cfb839a826a3e7adf7d220e6'),
 'checkpoint':pin(FIT/'fit/student_head.pt','9ceef5099ebd154e08a1c2c3784c4e06021d464e607474548763665b24f8f9e7'),
 'source_head':pin(FIT/'fit/student_head.onnx','f9b352ce4cedbbdd70c59f09b8a28080f7a696bfff464621719cc93e6c899797'),
 'normalization':pin(FIT/'fit/normalization.npz','1baf7a2918fe2322ad0ef9583a29902eb7c3d8f8d37000c6bb9fd502d6746079'),
 'training_manifest':pin(FIT/'training_frozen_inputs.json','1b9499a88b26b9feb5e1259d63284010e7521150c591a9815063cacc0251fa9c'),
 'training_request':pin(FIT/'training_request.json','e3dbc8c41433699326cd611ea97767974b6eb9ce5de2c7b1f678f83594054625'),
 'root_training_audit':pin(BASE/'direct_target_continuation_failure_audit_v1/results_v1/report.json','0accc9eea0a90158dbcfe38bd85aa04ae90ea88ef224da7ea4f86837672d516c'),
 'owner_failure_verification':pin(FIT/'owner_failure_verification.json'),
 'process_exit':pin(FIT/'fit_process/exit.json'),
 'process_absence':pin(FIT/'process_absence.json'),
 'saved_nine_comparison_audit':pin(BASE/'direct_target_continuation_failed_export_review_v1/report.json','4596c7707f5e0b39ab655df03a7829a1304432e75176e46425afbe11744abced'),
 'root_audit_source_review':pin(BASE/'direct_target_continuation_failure_audit_source_review_v1/review.json','85bb7b25974b3f1410a8cdcacd54e822f97a9e794c08e717266ae764767d3124'),
}
fit=read(subjects['fit_report']['path']);root=read(subjects['root_training_audit']['path'])
owner=read(subjects['owner_failure_verification']['path']);ex=read(subjects['process_exit']['path']);absence=read(subjects['process_absence']['path'])
assert root['evidence_audit_passed'] is True and root['optimization_evidence_verified'] is True
assert root['passed'] is False and root['export_qualified'] is False and root['canonical_cleared'] is False
assert root['checks']==64656 and root['failed_release_gates']==['ordinary_final_completed','report_export_pass','all_export_parity_within_selected_tolerance']
for k in ('task_model_calls','ORT_calls','optimizer_updates','native_steps'):assert root[k]==0
for key in ('fit_report','checkpoint','source_head','normalization','training_manifest','training_request'):
    p=subjects[key]['path'];assert root['input_sha256'][p]==subjects[key]['sha256'],key
assert fit['ordinary_final_step']==55000 and fit['additional_updates']==50000 and fit['restored_start_step']==5000
assert fit['optimization_completed'] is True and fit['final_export_diagnostics_completed'] is True
for key in ('completed','numerical_gate_passed','export_parity_passed'):assert fit[key] is False
assert fit['all_frozen_inputs_unchanged'] is True and fit['checkpoint_selection'] is False
assert fit['checkpoint_sha256']==subjects['checkpoint']['sha256'] and fit['onnx_sha256']==subjects['source_head']['sha256']
expected={'training_head_rows_attempted':705500000,'training_head_rows_returned':705500000,'diagnostic_torch_rows_attempted':460740,'diagnostic_torch_rows_returned':460740,'ORT_calls_attempted':601,'ORT_calls_returned':601}
assert fit['counters']==owner['counters']==expected
assert fit['export_parity']['maximum_preclamp_rad']==root['maximum_preclamp_export_error_rad']==1.5560187542007498e-5
assert fit['export_parity']['tolerance_rad']==1e-5 and fit['export_parity']['nonfinite_comparisons']==[]
assert owner['owner_evidence_verification_passed'] is True and owner['all_frozen_pins_exact'] is True
assert owner['report_sha256']==subjects['fit_report']['sha256'] and owner['exit_sha256']==subjects['process_exit']['sha256']
assert owner['numerical_gate_passed'] is False and owner['canonical_evaluation_cleared'] is False
assert ex['exit_known'] is True and ex['raw_python_exit_code']==ex['exit_code']==1 and ex['all_postrun_pins_exact'] is True
assert ex['automatic_retry'] is False and absence['any_present'] is False and absence['observed']==[]
assert set(absence['expected_pids'])=={ex['wrapper_pid'],ex['child_pid']}
request=read(subjects['training_request']['path'])
for key in ('centers','velocity_features','velocity_target','physical_manifest','pico','walk002'):
    p=Path(request['paths'][key]);subjects[key]=pin(p,root['input_sha256'][str(p)])
for value in subjects.values():assert sha(value['path'])==value['sha256']
report={'kind':'verified_training_and_dataset_only_failed_original_export_preserved',
 'passed':True,'training_review_pass':True,'dataset_review_pass':True,'optimization_evidence_verified':True,
 'fit_review_pass':False,'export_review_pass':False,'original_export_qualified':False,'canonical_evaluation_cleared':False,
 'ordinary_final_step':55000,'additional_updates':50000,'source_original_completed':False,
 'source_original_numerical_gate_passed':False,'source_original_export_parity_passed':False,
 'maximum_original_preclamp_error_rad':fit['export_parity']['maximum_preclamp_rad'],'unchanged_tolerance_rad':1e-5,
 'subjects':subjects,'direct_subject_sha256':{k:v['sha256'] for k,v in subjects.items()},
 'root_training_checks':root['checks'],'failed_original_release_gates':root['failed_release_gates'],
 'counters':expected,'all_direct_subject_hashes_verified':True,
 'source_sha256':sha(__file__),'model_calls':0,'native_steps':0,'optimizer_updates':0,
 'scope':['Verified ordinary55000 weights, restored optimizer/RNG/normalization, fixed data/schedule and completed50000 training updates through independent saved audit.',
          'Original FP32 export remains failed at the unchanged1e-5 rad numerical gate. This receipt authorizes neither that export nor any controller rollout.',
          'A separate numerical export and its actual runtime activation require their own qualified evidence.',
          'Final nominal and physical objectives improve; full velocity objective worsens22.3924 percent versus restored5000. No stability claim.']}
with (OUT/'review.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'training_review_pass':True,'dataset_review_pass':True,'export_review_pass':False,'review_sha256':sha(OUT/'review.json')}))
