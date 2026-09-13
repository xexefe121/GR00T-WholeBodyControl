import hashlib,json,sys
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent
NEW=OUT.parent
BASE=NEW/'independent_plant_pending_publication_saved_actual_v1'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
rp,lp=BASE/'request.json',BASE/'launch_receipt.json'
assert sha(rp)=='fba2b2841eab6a9f7697c472760e789224afd40ca71ca7c9c8692fd1b3251fdc'
assert sha(lp)=='63c8b002d6cd56cfc918cbc44f26b665cd0878240e4544c17ee30fac28cb685c'
r,l=read(rp),read(lp)
assert r['kind']=='independent_saved_clock_audit' and r['root_selected_saved_audit'] is True
for k in ('native_steps','model_calls','optimizer_updates','worker_processes'):assert r[k]==l[k]==0
assert l['automatic_retry'] is False and l['request_sha256']==sha(rp)
assert l['selected_single_saved_audit'] is True and l['final_concrete_review_required'] is True
review=read(r['source_review']['path'])
assert sha(r['source_review']['path'])==r['source_review']['sha256']==l['source_review_sha256']=='81c4090dd6035fc4b39064fd98e7412af87e2d0355a4a3abd260d19ffb4faae0'
assert review['passed'] is True and review['source_sha256']==r['source_sha256']
source=NEW/'independent_plant_pending_publication_saved_audit_v1/source_draft_v2'
assert len(r['source_sha256'])==17
for name,digest in r['source_sha256'].items():assert sha(source/name)==digest,name
helper_review=NEW/'independent_pending_saved_helper_root_review_v1/review.json'
assert sha(helper_review)=='c2e17f4e4eb28c8cdb4aa95e3deb9fb2cf8beb20cedff4b0cf382caee53b44b0'
assert read(helper_review)['passed'] is True
assert sha(BASE/'helper_preparation.json')=='20407fd5dc10b837e8c8e87b94a69e337b5e066ed52398ac392147693097af18'
prep=read(BASE/'helper_preparation.json')
assert prep['passed'] is True and prep['tests']==10 and prep['failures']==prep['errors']==prep['skips']==0
assert prep['audit_source_sha256']==r['source_sha256']
for name,digest in prep['helper_sha256'].items():assert sha(BASE/name)==digest,name
for name,digest in prep['evidence_sha256'].items():assert sha(BASE/name)==digest,name
assert sha(BASE/'preserved_saved_audit_template.ps1.txt')==prep['original_template_sha256']=='abd6286188c2df56ecc44aaf67ad6216696c46432faab62c0b6b46b69106c75f'
sys.path.insert(0,str(BASE))
from prepare_launch import render
assert (BASE/'run_audit_durable.ps1').read_text()==render(sha(rp))
pins=l['input_sha256'];assert len(pins)==3827 and len(r['input_sha256'])==3817
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
assert sha(roles['owner'])=='a51d7f88bff439a23c696ad48fa6cff5c8ba035cae58820cd1ac2c7f4ccf8b62'
assert sha(roles['run_report'])=='df8d562e18508b24372b81a7ecab6e8e0d53e195562078dc23ff6a45d54a89f6'
assert sha(roles['run_request'])=='2b538b03eb25de6d4c2ea3165561371af2415089957f4b9dd0bf9020a893118c'
assert sha(roles['launch_receipt'])=='c7cc875c5f4c1d36e9efef5a56c8091d638e66d48393a0086bb2b30681c4dc3e'
assert all(not (BASE/n).exists() for n in ('process_v1','results_v1','launch_clearance.json','owner_completion.json','dispatch.json'))
result=dict(passed=True,reviewer='root',reviewed_utc=datetime.now(timezone.utc).isoformat(),
 request_subject=dict(path=rp.as_posix(),sha256=sha(rp)),launch_receipt_subject=dict(path=lp.as_posix(),sha256=sha(lp)),
 selected_single_saved_audit=True,all_current_pins_exact=True,checked_input_pins=len(pins),request_subjects=len(r['input_sha256']),
 source_review_sha256=l['source_review_sha256'],helper_preparation_sha256=sha(BASE/'helper_preparation.json'),
 actual_durable_sha256=sha(BASE/'run_audit_durable.ps1'),writer_sha256=sha(__file__),
 reviewed_semantics=['Reviewed retry-aware v2 saved-only auditor, exact pending-BUSY clock run and final six stages',
 'hidden acquired-handle WSL child with create-new lock, complete pre/post pins and preserved raw exit',
 'positive evidence integrity does not require successful physical, command or timing outcome',
 'owner verifies exact request, comparisons ledger, zero actual model/native/update calls and process absence',
 'exact concrete review is required before dispatch; no automatic audit or native retry'],
 dispatch_performed=False,model_calls=0,ORT_calls=0,native_steps=0,optimizer_updates=0,behavioral_qualification=False)
path=OUT/'concrete_review.json'
with path.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(dict(path=path.as_posix(),sha256=sha(path),pins=len(pins),passed=True)))
