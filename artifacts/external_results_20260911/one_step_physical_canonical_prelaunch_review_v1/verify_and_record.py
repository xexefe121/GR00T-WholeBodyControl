import hashlib,json,re
from pathlib import Path
import numpy as np
N=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911');B=N/'one_step_physical_student_evaluation_v1';O=Path(__file__).parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(name,value):
    with (O/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def has(value,h):
    if isinstance(value,dict):return any(has(v,h) for v in value.values())
    if isinstance(value,list):return any(has(v,h) for v in value)
    return value==h
binding_path=B/'evaluation_binding.json';launch_path=B/'evaluation_process/launch_receipt.json'
assert sha(binding_path)=='eee281fc2d1d316367d31838140f5cbef25d44f62e965180354a55933647030c'
assert sha(launch_path)=='271a0d1e22bd1ea0eebe4c8ad31a7c96b630cfee152b39c1b6b19bde3436b0ff'
binding=read(binding_path);launch=read(launch_path);selected=read(B/'evaluation_reviews_selected.json')
witness_path=B/'head_witness/witness.npz';witness_report=B/'head_witness/report.json'
assert sha(witness_path)=='fa2dad8a24f43f577ec1120a86e41d7e76f6d09805cc0e637b14907c0a56e088'
assert sha(witness_report)=='9bc2c1f112ae321b9ef5a4a6661afe3c556381b34ee7857ad9cca52fce58e96d'
w=read(witness_report);completed=read(B/'witness_completion_verification.json');ex=read(B/'witness_process/exit.json')
assert w['pass_all'] is True and w['ordinary_final_step']==75000
assert w['expected_head_calls']==w['attempted_head_calls']==1
assert w['BFM_inference_calls']==w['physics_steps']==w['fit_head_calls_repeated']==0 and w['fitting_launched'] is False
assert w['control']==250 and w['source_frame']==261 and w['centers_row']==2038
assert w['exact_original_query250_inputs'] is True and w['pinned_WSL_runtime'] is True
assert w['onnxruntime_version']=='1.23.2' and w['execution_mode']=='ORT_SEQUENTIAL' and w['execution_provider']=='CPUExecutionProvider'
assert w['intra_op_threads']==w['inter_op_threads']==1
assert completed['passed'] is True and type(completed['all_launch_pins_unchanged']) is int and completed['all_launch_pins_unchanged']==1565
assert completed['exit_sha256']==sha(B/'witness_process/exit.json') and ex['exit_code']==0 and ex['error'] is None
assert w['head_sha256']==binding['head']['sha256']=='fb856003734acc0338586482968a7e31553a11826e549a8b486e0662e4934e81'
assert w['witness_sha256']==binding['first_export_witness']['sha256']==sha(witness_path)
assert binding['first_export_receipt']['sha256']==sha(witness_report)
with np.load(witness_path,allow_pickle=False) as z,np.load(binding['centers']['path'],allow_pickle=False) as c,np.load(binding['query250_labels']['path'],allow_pickle=False) as q:
    assert z['features'].dtype==np.float32 and z['features'].shape==(1069,)
    assert z['onnx_delta'].dtype==np.float32 and z['onnx_delta'].shape==(23,) and np.isfinite(z['onnx_delta']).all()
    assert z['features'].tobytes()==c['features'][2038].tobytes()==q['features'][0].tobytes()
    assert int(c['control'][2038])==int(q['control'][0])==int(z['control'])==250
    assert int(z['source_frame'])==261
    for name,key in [('head_sha256','head'),('centers_sha256','centers'),('query250_labels_sha256','query250_labels')]:
        assert str(z[name].item())==binding[key]['sha256']
    assert str(z['runtime_binary_sha256'].item())==w['runtime_binary_sha256']
assert selected['root_selected'] is True and selected['stage']=='evaluation' and selected['ordinary_final_step']==75000
assert selected['requested_main_controls']==binding['requested_main_controls']==launch['requested_main_controls']==1569
assert selected['conditional_hold_controls']==binding['conditional_hold_controls']==launch['conditional_hold_controls']==250
assert selected['separate_head_calls']==launch['expected_separate_head_calls']==0
assert binding['root_authorized_canonical_evaluation'] is True
assert binding['controller']=='original_unfiltered_raw_combined_action'
assert binding['filters_enabled'] is False and binding['compiled_preview_enabled'] is False and binding['hardware_authorized'] is False
assert len(binding['input_files'])==1566 and len(launch['input_hashes'])==1569
pins=dict(launch['input_hashes'])
for entry in binding['input_files']:assert pins[entry['path']]==entry['sha256']
for role,entry in binding['reviews'].items():
    assert sha(entry['path'])==entry['sha256'];review=read(entry['path']);assert review[entry['pass_field']] is True
    required={'fit':[binding['fit_report']['sha256']],'export':[binding['head']['sha256']],
        'source':[sha(B/'source_snapshot_v1/evaluate_physical_response_student.py')],
        'dataset':[binding[k]['sha256'] for k in ('centers','physical_manifest','physical_report','physical_collection_request')]}[role]
    assert all(has(review,h) for h in required),role
for key,digest in selected['selected_subject_sha256'].items():assert binding[key]['sha256']==digest
for name,digest in read(B/'source_freeze.json')['source_sha256'].items():assert sha(B/'source_snapshot_v1'/name)==digest
expected=list(read(B/'witness_process/launch_receipt.json')['exact_wsl_arguments'])
expected[-1]=expected[-1].replace('head_activation_witness.py','evaluate_physical_response_student.py')
assert launch['exact_wsl_arguments']==expected and launch['binding_sha256']==sha(binding_path)
run=(B/'evaluation_process/run.ps1').read_text();durable=(B/'evaluation_process/run_durable.ps1').read_text()
assert re.findall(r"'([^']*)'",run.split('$arguments = @(\n',1)[1].split('\n)',1)[0])==expected
for text in ['execution.lock','CreateNew','completed_controls -ne 1569','completed_controls -ne 250','physics_steps -ne 15690','physics_steps -ne 2500','quiet_standing_diagnostic_pass','actual_query250_ownexport_output_parity.json']:
    assert text in run,text
assert 'started.lock' in durable and 'CreateNew' in durable and '-WindowStyle Hidden' in durable
assert durable.index('$nativeHandle = $taskChild.Handle')<durable.index('$taskChild.WaitForExit()')
assert 'if ($null -eq $exitCode)' in durable
post=read(B/'witness_process/postrun_hashes.json');old=read(B/'witness_process/launch_receipt.json')
assert post==old['input_hashes']
for p,h in old['input_hashes'].items():
    if p in pins:assert pins[p]==h,p
    pins[p]=h
for p in [binding_path,launch_path,witness_path,witness_report,B/'witness_completion_verification.json',B/'witness_process/exit.json',B/'witness_process/postrun_hashes.json',B/'evaluation_reviews_selected.json']:
    pins[p.as_posix()]=sha(p)
for p,h in pins.items():assert sha(p)==h,p
for name in ['nominal','post_lifecycle_hold_5s','pilot_outcome.json','evaluation_process/started.lock','evaluation_process/execution.lock','evaluation_process/start.json','evaluation_process/child.json','evaluation_process/exit.json','evaluation_process/stdout.log','evaluation_process/stderr.log']:
    assert not (B/name).exists(),name
save('verification.json',dict(passed=True,input_sha256=pins,files_verified=len(pins),canonical_launch_pins=1569,
    witness_launch_pins=1565,source_files=25,witness_features_byteexact=True,witness_head_calls=1,
    reviewer_model_calls=0,reviewer_native_steps=0,reviewer_optimizer_updates=0))
subjects=dict(binding=dict(path=binding_path.as_posix(),sha256=sha(binding_path)),launch=dict(path=launch_path.as_posix(),sha256=sha(launch_path)),
    witness=dict(path=witness_path.as_posix(),sha256=sha(witness_path)),witness_report=dict(path=witness_report.as_posix(),sha256=sha(witness_report)))
for key in ('head','checkpoint','fit_report','training_manifest','centers','physical_manifest','physical_report','physical_collection_request'):subjects[key]=binding[key]
review=dict(kind='independent_ordinary75000_canonical_prelaunch_review',passed=True,source_review_pass=True,
    canonical_launch_review_pass=True,canonical_launch_authorized=True,ordinary_final_step=75000,subjects=subjects,
    final_fit_review=binding['reviews']['fit'],source_preparation_review=binding['reviews']['source'],
    verification=dict(path=(O/'verification.json').as_posix(),sha256=sha(O/'verification.json')),
    requested_main_controls=1569,conditional_hold_controls=250,source_controls=819,
    controller='original_unfiltered_raw_combined_action',filters_enabled=False,compiled_preview_enabled=False,
    additional_separate_head_calls=0,hardware_authorized=False,
    verdict='CLEAR for the root-selected ONE fresh canonical 1569-control lifecycle with conditional continuous 250-control hold, through the exact frozen durable launcher.',
    findings=[],limitations=['Source tracking, quiet standing and actual physical qualification remain outcome gates. Failed or partial trials must be preserved; no retry, checkpoint selection, deadline relaxation or history reset is authorized.'])
save('review.json',review)
print(json.dumps(dict(review_sha256=sha(O/'review.json'),verification_sha256=sha(O/'verification.json'),files_verified=len(pins))))
