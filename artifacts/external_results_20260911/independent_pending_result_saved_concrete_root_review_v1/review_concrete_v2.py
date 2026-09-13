import hashlib, json, sys, subprocess
from pathlib import Path
from datetime import datetime, timezone

OUT=Path(__file__).resolve().parent
NEW=OUT.parent
BASE=NEW/'independent_plant_pending_result_saved_actual_v1'
SOURCE=NEW/'independent_plant_pending_result_saved_audit_v1/source_draft_v2'
OLD=NEW/'independent_plant_pending_publication_saved_actual_v1'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

rp,lp=BASE/'request.json',BASE/'launch_receipt.json'
assert sha(rp)=='d5db2163820b0dd244b4d4a3de4ee4db4a324d8bdee3255048d5e35fc0346e51'
assert sha(lp)=='f1e55e3b436161feb90653545d2fc24675dd9df636309967294a7794e2a990ab'
r,l=read(rp),read(lp)
assert r['kind']=='independent_saved_clock_audit' and r['root_selected_saved_audit'] is True
for k in ('native_steps','model_calls','optimizer_updates','worker_processes'):assert r[k]==l[k]==0
assert l['automatic_retry'] is False and l['request_sha256']==sha(rp)
assert l['selected_single_saved_audit'] is True and l['final_concrete_review_required'] is True
review=read(r['source_review']['path'])
assert sha(r['source_review']['path'])==r['source_review']['sha256']==l['source_review_sha256']=='009d21d7dcc8b347da5d0f511ae2d2c24c22774eef84f8ba12f1e7101cb37a48'
assert review['passed'] is True and review['source_sha256']==r['source_sha256']
assert len(r['source_sha256'])==20
for name,digest in r['source_sha256'].items():assert sha(SOURCE/name)==digest,name
assert sha(BASE/'helper_preparation.json')=='6cc86cfd5976dfcd51a036b509bd9cd8879c85ca9ca8bfd375836b6e0089cd10'
prep=read(BASE/'helper_preparation.json')
assert prep['passed'] is True and prep['tests']==11 and prep['failures']==prep['errors']==prep['skips']==0
assert prep['audit_source_sha256']==r['source_sha256']
for name,digest in prep['helper_sha256'].items():assert sha(BASE/name)==digest,name
for name,digest in prep['evidence_sha256'].items():assert sha(BASE/name)==digest,name
assert sha(BASE/'preserved_saved_audit_template.ps1.txt')==sha(OLD/'preserved_saved_audit_template.ps1.txt')==prep['original_template_sha256']=='abd6286188c2df56ecc44aaf67ad6216696c46432faab62c0b6b46b69106c75f'
assert sha(BASE/'verify_completion.py')==sha(OLD/'verify_completion.py')=='2ad352000d8435b068ec2009b7bd5204a161feaa13fc96c4c0f8eac4d07a94e7'
# Production launcher differs only in reviewed source/review literals.
old=(OLD/'prepare_launch.py').read_text(encoding='utf-8-sig')
old=old.replace('independent_plant_pending_publication_saved_audit_v1','independent_plant_pending_result_saved_audit_v1').replace('independent_pending_publication_saved_root_review_v1','independent_pending_result_saved_audit_review_v1').replace('81c4090dd6035fc4b39064fd98e7412af87e2d0355a4a3abd260d19ffb4faae0','009d21d7dcc8b347da5d0f511ae2d2c24c22774eef84f8ba12f1e7101cb37a48')
assert old==(BASE/'prepare_launch.py').read_text(encoding='utf-8-sig')
sys.path.insert(0,str(BASE))
from prepare_launch import render
actual=(BASE/'run_audit_durable.ps1').read_text()
assert actual==render(sha(rp))
assert (BASE/'template_preview.ps1.txt').read_text()==render('a'*64)
assert actual==(BASE/'template_preview.ps1.txt').read_text().replace('a'*64,sha(rp))
assert 'param([Parameter(Mandatory=$true)][string]$LaunchReceiptSha256)' in actual
with (OUT/'root_helper_tests.log').open('x',encoding='utf-8') as log:
    test=subprocess.run([sys.executable,'-B','-m','unittest','test_launch','-v'],cwd=BASE,stdout=log,stderr=subprocess.STDOUT)
assert test.returncode==0
pins=l['input_sha256'];assert len(pins)==3842 and len(r['input_sha256'])==3832
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
for key,digest in {'owner':'968a2e8bb369a9168c0383503cb4775c1bcf87bf5482703d8c233ca3476017ae','run_report':'209b04cc616fce8b51ba76d7af593533f1099427da33c4b30fa90db0b8da3bff','run_request':'bfa3e9364ca648c810260b39ee86d4480a20a69340c399978338871f655d1f3e','launch_receipt':'8e9ef98e4bfcab9f8f7c3eaa9d6dda4ac3f6b3eb7151734d5dfb61398c0111b2'}.items():assert sha(roles[key])==digest,key
assert all(not (BASE/n).exists() for n in ('process_v1','results_v1','launch_clearance.json','owner_completion.json','dispatch.json'))
result=dict(passed=True,reviewer='root',reviewed_utc=datetime.now(timezone.utc).isoformat(),
 request_subject=dict(path=rp.as_posix(),sha256=sha(rp)),launch_receipt_subject=dict(path=lp.as_posix(),sha256=sha(lp)),
 selected_single_saved_audit=True,all_current_pins_exact=True,checked_input_pins=len(pins),request_subjects=len(r['input_sha256']),
 source_review_sha256=l['source_review_sha256'],helper_preparation_sha256=sha(BASE/'helper_preparation.json'),
 root_helper_tests=11,root_helper_tests_sha256=sha(OUT/'root_helper_tests.log'),actual_durable_sha256=sha(BASE/'run_audit_durable.ps1'),writer_sha256=sha(__file__),
 reviewed_semantics=['Retry-aware v2 with global worker iteration order and exact completed pending-result trial',
 'Original hidden acquired-handle wrapper and owner unchanged, mandatory dispatch argument preserved',
 'Exact 20 source files, four helpers, six saved stage records and every current input pin verified',
 'Evidence integrity remains separate from failed native bounds, missed commands and deadlines',
 'One saved-only audit; no actual worker, model, native integration or optimizer calls'],
 dispatch_performed=False,model_calls=0,ORT_calls=0,native_steps=0,optimizer_updates=0,behavioral_qualification=False)
with (OUT/'concrete_review.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2);stream.write('\n')
print(json.dumps(dict(path=(OUT/'concrete_review.json').as_posix(),sha256=sha(OUT/'concrete_review.json'),pins=len(pins),passed=True)))
