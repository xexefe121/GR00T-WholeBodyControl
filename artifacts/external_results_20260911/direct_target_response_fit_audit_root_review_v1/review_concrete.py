"""Root concrete binding check; metadata and hashes only, no task execution."""
from pathlib import Path
import hashlib,json,datetime

NEW=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')
BASE=NEW/'direct_target_response_balanced_fit_independent_v1'
OUT=Path(__file__).parent
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()
request=read(BASE/'audit_request.json'); launch=read(BASE/'launch_receipt.json')
assert sha(BASE/'audit_request.json')=='f23b1c471178a6dea64612c60ab37206c26848daa14172d588b39e96668c163d'
assert sha(BASE/'launch_receipt.json')=='4f007153b743de2bad653cdedf08094990d7f384b7bcd378f3c98edd3ff61dfb'
assert request['kind']==launch['kind']=='saved_response_balanced_warm_only'
assert request['experiment']==(NEW/'direct_target_causal_response_balanced_student_v2').as_posix()
assert launch['selected_single_saved_audit'] is True
assert request['automatic_retry'] is launch['automatic_retry'] is False
assert launch['request_sha256']==sha(BASE/'audit_request.json')
assert len(launch['input_sha256'])==36
for p,h in launch['input_sha256'].items(): assert sha(p)==h,(p,h)
review=read(request['source_review']['path'])
assert sha(request['source_review']['path'])==request['source_review']['sha256']==launch['source_review_sha256']
assert review['source_review_pass'] is review['helper_review_pass'] is True
assert len(request['source_sha256'])==11
for p,h in request['source_sha256'].items():
    assert review['source_sha256'][Path(p).name]==h==launch['input_sha256'][p]
for name,h in review['helper_sha256'].items(): assert sha(BASE/name)==h
owner=read(request['subjects']['owner_completion']['path'])
assert sha(request['subjects']['owner_completion']['path'])==launch['fit_owner_sha256']=='a6b75e9df7864e46a5ed0ca84efe05fc0fa2e020b25766d6077b3e6bc744b8d3'
for k in ('owner_verification_passed','accounting_passed','completion_passed','numerical_completion_passed','optimization_completed','processes_absent','raw_exit_known','all_postrun_pins_exact'): assert owner[k] is True,k
assert owner['raw_python_exit_code']==owner['exit_code']==0
roles=set(request['subjects'])-{'frozen_inputs','owner_completion'}
assert len(roles)==16
for role in roles:
    item=request['subjects'][role]
    assert sha(item['path'])==item['sha256']==owner['direct_subject_sha256'][role]
assert request['subjects']['frozen_inputs']==request['subjects']['training_manifest']
assert request['requires_completed_optimization_and_diagnostics'] is True
for k in ('task_model_calls','ORT_calls','gradient_calls','native_steps','optimizer_updates'): assert request[k]==0
assert not (BASE/'audit_process_v1').exists()
assert not (BASE/'results_v1').exists()
report=dict(concrete_review_pass=True,selected_single_saved_audit=True,reviewer='root',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),request_sha256=sha(BASE/'audit_request.json'),launch_receipt_sha256=sha(BASE/'launch_receipt.json'),source_review_sha256=launch['source_review_sha256'],fit_report_sha256=request['subjects']['fit_report']['sha256'],fit_owner_sha256=launch['fit_owner_sha256'],launch_pin_count=36,release_role_count=16,all_current_pins_exact=True,automatic_retry=False,sole_dispatch_owner='review_continuation',limitations=['Checks exact completed package; actual saved audit has not yet executed.'],task_model_calls=0,ORT_calls=0,gradient_calls=0,optimizer_updates=0,native_steps=0,writer_sha256=sha(__file__))
with (OUT/'concrete_review.json').open('x',encoding='utf-8') as f: json.dump(report,f,indent=2); f.write('\n')
print(json.dumps({'concrete_review_pass':True,'sha256':sha(OUT/'concrete_review.json'),'pins':36,'release_roles':16}))
