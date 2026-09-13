import argparse,hashlib,json,sys
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent
NEW=OUT.parent
BASE=NEW/'direct_target_response_saved_semantics_review_v1'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
parser=argparse.ArgumentParser()
parser.add_argument('--request-sha',required=True);parser.add_argument('--launch-sha',required=True)
parser.add_argument('--request-pins',type=int,required=True);parser.add_argument('--launch-pins',type=int,required=True)
args=parser.parse_args()
rp,lp=BASE/'request.json',BASE/'launch_receipt.json'
assert sha(rp)==args.request_sha and sha(lp)==args.launch_sha
r,l=read(rp),read(lp)
assert r['kind']=='one_pure_saved_causal71000_runtime_semantics_and_fixed_maps'
assert r['context_condition']=='causal' and r['public_features']==1323
assert r['original_requested_controls']==1569 and r['conditional_continuous_hold_controls']==250
assert r['model_calls']==r['native_steps']==r['BFM_calls']==r['optimizer_updates']==r['replans']==0
assert l['model_calls']==l['native_steps']==l['optimizer_updates']==0 and l['automatic_retry'] is False
assert l['request_sha256']==sha(rp) and l['final_concrete_review_required'] is True
assert sha(OUT/'review.json')==r['source_review_sha256']==l['source_review_sha256']=='3bf735731db9341ff63d01f8c7e4ad8eb1767f39d264ea34624d8babcb337671'
prep=read(BASE/'source_preparation.json');stage=read(BASE/'stage_source_preparation.json')
assert read(OUT/'review.json')['source_sha256']==prep['source_sha256']==stage['audit_source_sha256']
for n,h in prep['source_sha256'].items():assert sha(BASE/n)==h
for n,h in stage['helper_sha256'].items():assert sha(BASE/n)==h
assert stage['passed'] is True and stage['synthetic_tests_passed'] is True and stage['tests_run']==56
assert sha(BASE/'preserved_run_audit_template.ps1.txt')==stage['preserved_template_sha256']==sha(stage['original_template']['path'])
sys.path.insert(0,str(BASE))
from prepare_audit_stage import render
assert (BASE/'run_audit_durable.ps1').read_text()==render(sha(rp))
pins=l['input_sha256']
assert len(pins)==args.launch_pins and len(r['input_sha256'])==args.request_pins
normalized={}
for p,h in pins.items():
    k=p.replace('\\','/').casefold()
    assert k not in normalized or normalized[k]==h
    normalized[k]=h
for p,h in r['input_sha256'].items():assert normalized[p.replace('\\','/').casefold()]==h
for p,h in pins.items():assert sha(p)==h,p
p=r['paths']
assert sha(p['owner'])==r['owner_sha256']=='653fe9a1f06aa7138dad468891f823b71efa42fc9ed0ce11a83851b9c9c85fcc'
assert sha(p['main_physics'])=='d1ad17ef99dd0294b5ccc402c6073536b6e49ecafeea23a86dd1d004b7ea51eb'
assert sha(p['root_intent'])=='acbcad444f1a8a6ba58c2aac22aeb82bfa6615afa7bcd0317abb13fbd8d66a95'
assert sha(p['main_trace'])=='78b2f9692b6de5d5d8b5310f9ac4e4803cc43a45532bb4ae0e79bea94f604801'
owner=read(p['owner']);physics=read(p['main_physics']);intent=read(p['root_intent'])
assert owner['owner_completion_accounting_passed'] is True and owner['raw_python_exit_code']==0 and owner['diagnostic_exit_code']==2
assert owner['process_absence']['wrapper_absent'] is True and owner['process_absence']['child_absent'] is True
assert physics['recorded_trace_reproduced_through_last_sample'] is True and physics['physics_steps']==2879
assert all(physics['original_trace_comparison'].values()) and physics['private_replay_steps_beyond_recorded_prefix']==0
assert intent['recorded_controls']==288 and intent['requested_controls']==1569
assert intent['full_lifecycle_source_intent_pass'] is False and intent['requested_segment_quiet_pass'] is False
assert 'hold_trace' not in p and 'main_failure' in p
assert all(not (BASE/n).exists() for n in ('process_v1','results_v1','launch_clearance.json','owner_completion.json','dispatch.json'))
result=dict(passed=True,reviewer='root',reviewed_utc=datetime.now(timezone.utc).isoformat(),
 request_subject=dict(path=rp.as_posix(),sha256=sha(rp)),launch_receipt_subject=dict(path=lp.as_posix(),sha256=sha(lp)),
 selected_single_saved_audit=True,all_current_pins_exact=True,checked_input_pins=len(pins),request_subjects=len(r['input_sha256']),
 source_review_sha256=sha(OUT/'review.json'),stage_preparation_sha256=sha(BASE/'stage_source_preparation.json'),
 actual_durable_sha256=sha(BASE/'run_audit_durable.ps1'),writer_sha256=sha(__file__),
 reviewed_semantics=['Exact causal71000 request/source/helper maps and actual failure trace.',
 'Full1569 and conditional250 unchanged; completed root2879-step exact physical replay plus same-trace false intent verdicts.',
 'CreateNew lock, acquired hidden WSL child handle, known raw exit, post hashes and process absence.',
 'Concrete request/launcher review bound before and after; single saved-only audit, no numerical model or native calls.'],
 dispatch_performed=False,model_calls=0,ORT_calls=0,native_steps=0,optimizer_updates=0,behavioral_qualification=False)
path=OUT/'concrete_review.json'
with path.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(dict(path=path.as_posix(),sha256=sha(path),pins=len(pins),passed=True)))
