import hashlib,json,sys
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent
NEW=OUT.parent
BASE=NEW/'direct_target_context_saved_semantics_review_v1'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
rp,lp=BASE/'request.json',BASE/'launch_receipt.json'
assert sha(rp)=='8694708a3370fdd26e7f6f4dccb321a7c5afa1ba381a7f9df670511d953e6dcf'
assert sha(lp)=='7e3b11afef444d1a627e0eb11d6bda6e1b717a2d0f89d902a2082d3c1cb16c18'
r,l=read(rp),read(lp)
assert r['context_condition']=='causal' and r['public_features']==1323
assert r['original_requested_controls']==1569 and r['conditional_continuous_hold_controls']==250
assert r['model_calls']==r['native_steps']==r['BFM_calls']==r['optimizer_updates']==r['replans']==0
assert l['model_calls']==l['native_steps']==l['optimizer_updates']==0 and l['automatic_retry'] is False
assert l['request_sha256']==sha(rp) and l['final_concrete_review_required'] is True
assert sha(OUT/'review.json')==r['source_review_sha256']==l['source_review_sha256']=='73517c40584eeda0f7982f0f0653da3aecd949fc28dda85b091d9431799db89d'
prep=read(BASE/'source_preparation.json');stage=read(BASE/'stage_source_preparation.json')
assert read(OUT/'review.json')['source_sha256']==prep['source_sha256']==stage['audit_source_sha256']
for n,h in prep['source_sha256'].items():assert sha(BASE/n)==h
for n,h in stage['helper_sha256'].items():assert sha(BASE/n)==h
for n,h in stage['evidence_sha256'].items():assert sha(BASE/n)==h
assert stage['passed'] is True and stage['synthetic_tests_passed'] is True and stage['tests_run']==13
assert stage['failures']==stage['errors']==stage['skips']==0
assert sha(BASE/'preserved_run_audit_template.ps1.txt')==stage['preserved_template_sha256']==sha(stage['original_template']['path'])
sys.path.insert(0,str(BASE))
from prepare_audit_stage import render
assert (BASE/'run_audit_durable.ps1').read_text()==render(sha(rp))
pins=l['input_sha256']
assert len(pins)==1658 and len(r['input_sha256'])==101
normalized={}
for p,h in pins.items():
    k=p.replace('\\','/').casefold()
    assert k not in normalized or normalized[k]==h
    normalized[k]=h
for p,h in r['input_sha256'].items():assert normalized[p.replace('\\','/').casefold()]==h
for p,h in pins.items():assert sha(p)==h,p
p=r['paths']
assert sha(p['owner'])==r['owner_sha256']=='c1e578569bd9f7e1bb0c86342f196440573ecbfc2f574e6fa3dce7fafc32f97e'
assert sha(p['main_physics'])=='fccc2ec215c8536529161eb98982038232ac9116bcfdbec685233807b60f0753'
assert sha(p['main_trace'])=='589cbab8051ab9c1257049d62098248e7866b10e6c0eb085e73cbaf32d79082d'
owner=read(p['owner']);physics=read(p['main_physics'])
assert owner['owner_completion_accounting_passed'] is True and owner['raw_python_exit_code']==0 and owner['diagnostic_exit_code']==2
assert owner['process_absence']['wrapper_absent'] is True and owner['process_absence']['child_absent'] is True
assert physics['recorded_trace_reproduced_through_last_sample'] is True and physics['physics_steps']==3021
assert all(physics['original_trace_comparison'].values()) and physics['private_replay_steps_beyond_recorded_prefix']==0
assert 'hold_trace' not in p and 'main_failure' in p
assert all(not (BASE/n).exists() for n in ('process_v1','results_v1','launch_clearance.json','owner_completion.json','dispatch.json'))
result=dict(passed=True,reviewer='root',reviewed_utc=datetime.now(timezone.utc).isoformat(),
 request_subject=dict(path=rp.as_posix(),sha256=sha(rp)),launch_receipt_subject=dict(path=lp.as_posix(),sha256=sha(lp)),
 selected_single_saved_audit=True,all_current_pins_exact=True,checked_input_pins=len(pins),request_subjects=101,
 source_review_sha256=sha(OUT/'review.json'),stage_preparation_sha256=sha(BASE/'stage_source_preparation.json'),
 actual_durable_sha256=sha(BASE/'run_audit_durable.ps1'),writer_sha256=sha(__file__),
 reviewed_semantics=['exact causal request and source hash map','CreateNew lock and preserved raw exit',
 'concrete request/launcher review bound before and after execution','hidden WSL child with acquired handle',
 'single saved-only audit with existing NumPy/SciPy runtime pins','owner checks failure versus evidence success, complete outputs and PID absence'],
 dispatch_performed=False,model_calls=0,ORT_calls=0,native_steps=0,optimizer_updates=0,behavioral_qualification=False)
path=OUT/'concrete_review.json'
with path.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(dict(path=path.as_posix(),sha256=sha(path),pins=len(pins),passed=True)))
