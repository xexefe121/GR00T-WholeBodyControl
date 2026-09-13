import hashlib,json,re
from pathlib import Path
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
binding_path=B/'witness_binding.json';launch_path=B/'witness_process/launch_receipt.json'
assert sha(binding_path)=='de2625d2128dd286f950bd5791c58eaa81d6b4f124b8af7ce416a658bd942782'
assert sha(launch_path)=='214915acab2dfc64a555b9247a5ffc61d6ad7c16db4e1663dc065df3fbc886d7'
binding=read(binding_path);launch=read(launch_path);selected=read(B/'witness_reviews_selected.json')
assert selected['root_selected'] is True and selected['stage']=='witness'
assert selected['requested_main_controls']==selected['conditional_hold_controls']==0 and selected['separate_head_calls']==1
assert sha(Path(selected['root_selection_path']))==selected['root_selection_sha256']
assert binding['ordinary_final_step']==selected['ordinary_final_step']==75000
assert binding['root_authorized_single_head_witness'] is True and binding['expected_head_calls']==1
for name in ('BFM_inference_authorized','physics_authorized','fitting_authorized','hardware_authorized'):assert binding[name] is False
assert launch['expected_separate_head_calls']==1 and launch['requested_main_controls']==launch['conditional_hold_controls']==0
assert launch['binding_sha256']==sha(binding_path)
assert len(binding['input_files'])==1562 and len(launch['input_hashes'])==1565
pins=dict(launch['input_hashes'])
for entry in binding['input_files']:assert pins[entry['path']]==entry['sha256']
for role,entry in binding['reviews'].items():
    assert sha(entry['path'])==entry['sha256']
    review=read(entry['path']);assert review[entry['pass_field']] is True,role
    required={'fit':[binding['fit_report']['sha256']],'export':[binding['head']['sha256']],
        'dataset':[binding[k]['sha256'] for k in ('centers','physical_manifest','physical_report','physical_collection_request')],
        'source':[sha(B/'source_snapshot_v1/head_activation_witness.py')]}[role]
    assert all(has(review,h) for h in required),role
for key,digest in selected['selected_subject_sha256'].items():assert binding[key]['sha256']==digest
assert binding['head']['sha256']=='fb856003734acc0338586482968a7e31553a11826e549a8b486e0662e4934e81'
assert binding['checkpoint']['sha256']=='9f31d74855c28a57231c51e0a652a5f2eeeaac87a3c5aff032d59ba5ca981e34'
assert binding['fit_report']['sha256']=='5153ef3e193059062cf071b2b82c1961f7465683e8d8c9e5471dd89a27b25a1b'
for name,digest in read(B/'source_freeze.json')['source_sha256'].items():assert sha(B/'source_snapshot_v1'/name)==digest
linux='/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/one_step_physical_student_evaluation_v1'
expected=['-d','Ubuntu-22.04','--cd','/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof','--','bash',
    '/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh',
    'env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','PYTHONPATH='+linux+'/source_snapshot_v1',
    '/mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python',linux+'/source_snapshot_v1/head_activation_witness.py']
assert launch['exact_wsl_arguments']==expected
run=(B/'witness_process/run.ps1').read_text();durable=(B/'witness_process/run_durable.ps1').read_text()
argblock=run.split('$arguments = @(\n',1)[1].split('\n)',1)[0]
assert re.findall(r"'([^']*)'",argblock)==expected
assert 'execution.lock' in run and 'CreateNew' in run and 'Existing attempt output must be preserved' in run
assert 'started.lock' in durable and 'CreateNew' in durable and '-WindowStyle Hidden' in durable
assert durable.index('$nativeHandle = $taskChild.Handle')<durable.index('$taskChild.WaitForExit()')
assert "if ($null -eq $exitCode)" in durable and 'postrun_hashes.json' in durable
for p,h in pins.items():assert sha(p)==h,p
for p in [binding_path,launch_path,B/'witness_reviews_selected.json',Path(selected['root_selection_path'])]:pins[p.as_posix()]=sha(p)
for name in ('head_witness','witness_process/started.lock','witness_process/execution.lock','witness_process/start.json','witness_process/child.json','witness_process/exit.json','witness_process/stdout.log','witness_process/stderr.log'):
    assert not (B/name).exists(),name
save('verification.json',dict(passed=True,input_sha256=pins,launch_pins=1565,files_verified=len(pins),source_files=25,
    exact_wsl_arguments=expected,reviewer_model_calls=0,reviewer_native_steps=0,reviewer_optimizer_updates=0))
subjects={p:dict(path=p,sha256=sha(p)) for p in [str(binding_path),str(launch_path),str(B/'witness_reviews_selected.json')]}
for key in ('head','checkpoint','fit_report','training_manifest','centers','physical_manifest','physical_report','physical_collection_request'):subjects[key]=binding[key]
review=dict(kind='independent_ordinary75000_head_witness_prelaunch_review',passed=True,source_review_pass=True,
    witness_launch_review_pass=True,witness_launch_authorized=True,ordinary_final_step=75000,subjects=subjects,
    direct_subject_sha256={p:h for p,h in pins.items() if p in (binding_path.as_posix(),launch_path.as_posix())},
    final_fit_review=binding['reviews']['fit'],source_preparation_review=binding['reviews']['source'],
    verification=dict(path=(O/'verification.json').as_posix(),sha256=sha(O/'verification.json')),
    expected_head_calls=1,BFM_calls=0,native_steps=0,fitting_authorized=False,canonical_launch_authorized=False,
    verdict='CLEAR for exactly one root-selected batch-one WSL head activation witness through the frozen durable launcher.',
    findings=[],limitations=['Canonical binding and launcher review remain required after a successful saved witness. No additional inference or physics was run by reviewer.'])
save('review.json',review)
print(json.dumps(dict(review_sha256=sha(O/'review.json'),verification_sha256=sha(O/'verification.json'),files_verified=len(pins))))
