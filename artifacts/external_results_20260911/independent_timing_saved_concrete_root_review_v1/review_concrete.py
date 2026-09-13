"""Review the completed instrumented clock's single saved-only audit packet."""
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
OUT=Path(__file__).resolve().parent
NEW=OUT.parent
BASE=NEW/'independent_plant_timing_saved_actual_v1'
SOURCE=NEW/'independent_plant_timing_saved_audit_v1/source_draft_v2'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def normalized(pins):
    result={}
    for path,digest in pins.items():
        key=path.replace('\\','/').casefold()
        assert key not in result or result[key]==digest
        result[key]=digest
    return result
rp,lp=BASE/'request.json',BASE/'launch_receipt.json'
assert sha(rp)=='dd23d2ee96356e0381d384cd8839c8aaf72cada30f9ab6503ddf3b8233706a54'
assert sha(lp)=='4559e431a0ca69fc714e08fc583098f030faf8097306d1b865ae1463fdf09374'
r,l=read(rp),read(lp)
assert r['kind']=='independent_saved_clock_audit' and r['root_selected_saved_audit'] is True
for key in ('native_steps','model_calls','optimizer_updates','worker_processes'):assert r[key]==l[key]==0
assert l['automatic_retry'] is False and l['request_sha256']==sha(rp)
assert l['selected_single_saved_audit'] is True and l['final_concrete_review_required'] is True
review=read(r['source_review']['path'])
assert sha(r['source_review']['path'])==r['source_review']['sha256']==l['source_review_sha256']=='3615d4c6df567cb917bfd7d2835d1e5d4c4db76582bbac66f2497536c0399cf3'
assert review['passed'] is review['source_review_pass'] is True
assert review['source_sha256']==r['source_sha256'] and len(r['source_sha256'])==25
for name,digest in r['source_sha256'].items():assert sha(SOURCE/name)==digest,name
prep_path=BASE/'helper_preparation.json'
assert sha(prep_path)=='09d2e33299a5774876e3695255039bee020d4cd865d6fb8fd9894664b74f6945'
prep=read(prep_path);assert prep['passed'] is True and prep['audit_source_sha256']==r['source_sha256']
helper_review_path=NEW/'independent_timing_saved_audit_root_review_v1/launch_helpers_review.json'
assert sha(helper_review_path)=='e1fb37c1d1833ae263b9c9c37f56a166863747440792eeb8e5a3922525c8140d'
helper_review=read(helper_review_path)
assert helper_review['passed'] is helper_review['source_review_pass'] is True
package=next(p for p in helper_review['packages'] if p['package']==BASE.name)
assert package['helper_preparation_sha256']==sha(prep_path)
for group in ('helper_sha256','evidence_sha256'):
    for name,digest in prep[group].items():assert sha(BASE/name)==digest,name
assert sha(BASE/'verify_completion.py')=='2ad352000d8435b068ec2009b7bd5204a161feaa13fc96c4c0f8eac4d07a94e7'
sys.path.insert(0,str(BASE))
from prepare_launch import render
actual=(BASE/'run_audit_durable.ps1').read_text()
assert actual==render(sha(rp))
assert (BASE/'template_preview.ps1.txt').read_text()==render('a'*64)
assert actual==(BASE/'template_preview.ps1.txt').read_text().replace('a'*64,sha(rp))
assert 'param([Parameter(Mandatory=$true)][string]$LaunchReceiptSha256)' in actual
pins=l['input_sha256'];assert len(pins)==3871 and len(r['input_sha256'])==3861
normal=normalized(pins)
for path,digest in r['input_sha256'].items():assert normal[path.replace('\\','/').casefold()]==digest
for path,digest in pins.items():assert sha(path)==digest,path
assert len(r['stage_records'])==6
for entry in r['stage_records']:assert sha(entry['path'])==entry['sha256']
for key,digest in {
    'owner':'d51bc2b9aef2641dcb70edfb395c61c3345b480b387071601de665d4a9576886',
    'run_report':'5c8bebc02cbdfbb7a83849b13aa65114aacc2a672b8382c014f65d67857aad49',
    'run_request':'005fcf08f9a36bc34315eddfbe506432181d6ce5c2c5548b60153d35dcfaff80',
    'launch_receipt':'214221511d1a722061c4426317cb3555b3437b08377a137ca1e8b36bbad4db08'}.items():
    assert sha(r['roles'][key])==digest,key
for name in ('process_v1','results_v1','launch_clearance.json','owner_completion.json','dispatch.json'):
    assert not (BASE/name).exists(),name
result=dict(passed=True,reviewer='root',reviewed_utc=datetime.now(timezone.utc).isoformat(),
    request_subject=dict(path=rp.as_posix(),sha256=sha(rp)),launch_receipt_subject=dict(path=lp.as_posix(),sha256=sha(lp)),
    selected_single_saved_audit=True,all_current_pins_exact=True,checked_input_pins=len(pins),request_subjects=len(r['input_sha256']),
    source_review_sha256=l['source_review_sha256'],helper_preparation_sha256=sha(prep_path),
    exact_prior_helper_review_sha256=sha(helper_review_path),actual_durable_sha256=sha(BASE/'run_audit_durable.ps1'),
    writer_sha256=sha(__file__),reviewed_semantics=[
      'Actual completed instrumented clock, 25 reviewed source modules and six saved stage records.',
      'Exact reviewed hidden captured-handle supervisor, mandatory literal receipt argument, no retry.',
      'Original fixed epoch, native ledger and sidecars remain immutable; evidence integrity is separate from behavior.',
      'Saved numerical checks only, zero model/native/optimizer/worker execution.'],
    dispatch_performed=False,model_calls=0,ORT_calls=0,native_steps=0,optimizer_updates=0,behavioral_qualification=False)
with (OUT/'concrete_review.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2);stream.write('\n')
print(json.dumps({'passed':True,'pins':len(pins),'review_sha256':sha(OUT/'concrete_review.json')}))
