import hashlib,json,sys
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent
NEW=OUT.parent
BASE=NEW/'independent_plant_clock_saved_actual_v1'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
rp,lp=BASE/'request.json',BASE/'launch_receipt.json'
assert sha(rp)=='a4f6ddd9a8468c667e3de2b533d1f8098c7b93548c1c89de82909e839f30713f'
assert sha(lp)=='2579e1c45cc7686934fe344e8682ba8485b8aca4ad41cf1a098766fbd3c0bdf0'
r,l=read(rp),read(lp)
assert r['kind']=='independent_saved_clock_audit' and r['root_selected_saved_audit'] is True
for k in ('native_steps','model_calls','optimizer_updates','worker_processes'):assert r[k]==l[k]==0
assert l['automatic_retry'] is False and l['request_sha256']==sha(rp)
assert l['selected_single_saved_audit'] is True and l['final_concrete_review_required'] is True
review=read(r['source_review']['path'])
assert sha(r['source_review']['path'])==r['source_review']['sha256']==l['source_review_sha256']=='278c509418ef1c7a1be655693a840217dd8c1894a6a6c33f0757a712ad8b9ac7'
assert review['passed'] is True and review['source_sha256']==r['source_sha256']
source=NEW/'independent_plant_clock_saved_root_review_v1/source_audit_v3'
assert len(r['source_sha256'])==14
for name,digest in r['source_sha256'].items():assert sha(source/name)==digest,name
prep=read(BASE/'helper_preparation.json')
assert prep['passed'] is True and prep['tests']==10 and prep['failures']==prep['errors']==prep['skips']==0
assert prep['audit_source_sha256']==r['source_sha256']
for name,digest in prep['helper_sha256'].items():assert sha(BASE/name)==digest,name
for name,digest in prep['evidence_sha256'].items():assert sha(BASE/name)==digest,name
assert sha(BASE/'preserved_saved_audit_template.ps1.txt')==prep['original_template_sha256']=='abd6286188c2df56ecc44aaf67ad6216696c46432faab62c0b6b46b69106c75f'
sys.path.insert(0,str(BASE))
from prepare_launch import render
assert (BASE/'run_audit_durable.ps1').read_text()==render(sha(rp))
pins=l['input_sha256'];assert len(pins)==3802 and len(r['input_sha256'])==3792
normalized={}
for p,h in pins.items():
    key=p.replace('\\','/').casefold()
    assert key not in normalized or normalized[key]==h
    normalized[key]=h
for p,h in r['input_sha256'].items():assert normalized[p.replace('\\','/').casefold()]==h
for p,h in pins.items():assert sha(p)==h,p
assert len(r['stage_records'])==6
for sub in r['stage_records']:assert sha(sub['path'])==sub['sha256']
roles=r['roles']
assert sha(roles['owner'])=='3bf9f7ee97c7feb74434f1f928a33bb4827e62420a8f68f12f49ce5b73dd816e'
assert sha(roles['run_report'])=='b63896a22b00a032f9d1ac5e9fcb420d796778607374d11e4a1ad7fec3ec52fc'
assert sha(roles['run_request'])=='0c22640dfb617a20c31aff7356434a6253096b0fc4cbed38aae64faeb14e75eb'
assert sha(roles['launch_receipt'])=='f3eb4aca76e45b56761a95017090359a4de21d6edd1f3d7cacc71754158f1713'
assert all(not (BASE/n).exists() for n in ('process_v1','results_v1','launch_clearance.json','owner_completion.json','dispatch.json'))
result=dict(passed=True,reviewer='root',reviewed_utc=datetime.now(timezone.utc).isoformat(),
 request_subject=dict(path=rp.as_posix(),sha256=sha(rp)),launch_receipt_subject=dict(path=lp.as_posix(),sha256=sha(lp)),
 selected_single_saved_audit=True,all_current_pins_exact=True,checked_input_pins=len(pins),request_subjects=len(r['input_sha256']),
 source_review_sha256=l['source_review_sha256'],helper_preparation_sha256=sha(BASE/'helper_preparation.json'),
 actual_durable_sha256=sha(BASE/'run_audit_durable.ps1'),writer_sha256=sha(__file__),
 reviewed_semantics=['unchanged reviewed v3 saved-only auditor, exact corrected clock run and final six stages',
 'hidden acquired-handle WSL child with create-new lock, complete pre/post pins and preserved raw exit',
 'positive evidence integrity does not require successful physical, command or timing outcome',
 'owner verifies exact request, comparisons ledger, zero actual model/native/update calls and process absence',
 'exact concrete review is required before dispatch; no automatic audit or native retry'],
 dispatch_performed=False,model_calls=0,ORT_calls=0,native_steps=0,optimizer_updates=0,behavioral_qualification=False)
path=OUT/'concrete_review.json'
with path.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(dict(path=path.as_posix(),sha256=sha(path),pins=len(pins),passed=True)))
