"""Check one actual clock packet using metadata and file hashes only."""
import argparse,hashlib,json,sys
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;NEW=OUT.parent
BASE=NEW/'independent_plant_pending_result_v1';OLD=NEW/'independent_plant_process_clock_v1'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
parser=argparse.ArgumentParser()
for name in ('request-sha','launch-sha'):parser.add_argument('--'+name,required=True)
for name in ('request-pins','launch-pins'):parser.add_argument('--'+name,type=int,required=True)
a=parser.parse_args();rp=BASE/'clock_request.json';lp=BASE/'clock_process/launch_receipt.json'
assert sha(rp)==a.request_sha and sha(lp)==a.launch_sha
r,l=read(rp),read(lp);o,ol=read(OLD/'clock_request.json'),read(OLD/'clock_process/launch_receipt.json')
assert sha(OLD/'clock_request.json')=='d6a6c3332039046c77ff9c93eaedd0694f0ac3c8487a474760c25646e75ba91a'
assert sha(OLD/'clock_process/launch_receipt.json')=='4ecd884ea49e2b307febb63967c222946c9ef6fdce73a8b52c59571b9a383c80'
unchanged=('requested_controls','main_controls','hold_controls','native_step_budget','serialization_budget','model_inference_calls','optimizer_updates','other_oracle_native_steps','epoch_lead_ns','epoch_rebase_allowed','debt_abort_steps','elapsed_abort_ns','native_bundle','expected_model_sha256','command_table_sha256')
assert all(r[k]==o[k] for k in unchanged)
assert (r['requested_controls'],r['main_controls'],r['hold_controls'],r['native_step_budget'],r['serialization_budget'])==(1819,1569,250,18190,4)
assert r['model_inference_calls']==r['optimizer_updates']==r['other_oracle_native_steps']==0
assert r['epoch_lead_ns']==200000000 and r['epoch_rebase_allowed'] is False
assert r['debt_abort_steps']==100 and r['elapsed_abort_ns']==60000000000
assert r['watchdog_budgets']==dict(setup_ns=240000000000,plant_ns=120000000000,preservation_ns=180000000000,outer_timeout_seconds=555,outer_kill_grace_seconds=5)
assert r['outer_process_timeout_seconds']==555 and r['hardware_authorized'] is l['hardware_authorized'] is False
assert r['execution_selected'] is l['execution_selected'] is l['automatic_retry'] is False
assert l['final_clearance_required'] is True and l['request_path']==rp.as_posix() and l['request_sha256']==sha(rp)
assert l['requested_native_steps']==18190 and l['requested_serializations']==4
assert all(r['roles'][k]==v for k,v in o['roles'].items())
for key,name in [('output_directory','run'),('stage_directory','stage_receipts'),('source_directory','source_draft_v1')]:assert Path(r[key]).resolve()==(BASE/name).resolve()
assert r['input_epoch']==3 and r['run_id']=='query250-recorded-clock-pending-result-BUSY-v1'
pins=l['input_hashes'];assert len(pins)==a.launch_pins and len(r['input_files'])==a.request_pins
assert a.launch_pins==a.request_pins+3
assert len({Path(p).resolve().as_posix().lower() for p in pins})==len(pins)
for item in r['input_files']:assert pins[item['path']]==item['sha256']
for p,h in pins.items():assert sha(p)==h,p
external=0
for p,h in ol['input_hashes'].items():
    if not Path(p).resolve().is_relative_to(OLD.resolve()):assert pins[p]==h;external+=1
assert external==3689
helper_path=NEW/'independent_pending_result_helper_root_review_v1/review.json'
assert sha(helper_path)=='b5cb850fb0ea2349f96980b046db6d480d53cdb2947b704d937dd0a2101c6e9a'
helper=read(helper_path);assert helper['passed'] is helper['helper_review_pass'] is True
prep=read(BASE/'source_preparation.json');hp=read(BASE/'helper_preparation.json')
assert sha(BASE/'helper_preparation.json')==helper['helper_preparation_subject']['sha256']
assert helper['helper_sha256']==hp['helper_sha256']
for n,h in hp['helper_sha256'].items():assert sha(BASE/n)==h==pins[(BASE/n).as_posix()]
producer=read(r['roles']['pending_source_review'])
assert sha(r['roles']['pending_source_review'])==hp['producer_review_sha256']=='b017dbb753fe2c8a0370288de01eb9e52275e2e5a365b54582da20cb15700640'
assert producer['passed'] is producer['source_review_pass'] is True
assert producer['source_sha256']==prep['source_sha256']==hp['producer_source_sha256']==helper['producer_source_sha256']
assert len(prep['source_sha256'])==24
for n,h in prep['source_sha256'].items():assert sha(BASE/'source_draft_v1'/n)==h==pins[(BASE/'source_draft_v1'/n).as_posix()]
ap=read(r['roles']['saved_stage_audit_preparation']);ar=read(r['roles']['saved_stage_audit_source_review'])
assert sha(r['roles']['saved_stage_audit_preparation'])==hp['auditor_preparation_sha256']=='09bf03da9bae5285ccc24cabd77a83718e06485dbc267a50d047deceea1f3d91'
assert sha(r['roles']['saved_stage_audit_source_review'])==hp['auditor_review_sha256']=='009d21d7dcc8b347da5d0f511ae2d2c24c22774eef84f8ba12f1e7101cb37a48'
assert ar['passed'] is ar['source_review_pass'] is True and ar['source_sha256']==ap['source_sha256']==hp['auditor_source_sha256']==helper['auditor_source_sha256']
assert len(ap['source_sha256'])==20
for n,h in ap['source_sha256'].items():assert sha(NEW/'independent_plant_pending_result_saved_audit_v1/source_draft_v2'/n)==h
sys.path.insert(0,str(BASE))
from prepare_concrete_packet import validate_retry_contract,check_subject
from prepare_clock_stage import arguments,texts
validate_retry_contract(r)
check_subject(ar['producer_preparation'],BASE/'source_preparation.json',sha(BASE/'source_preparation.json'))
check_subject(ar['producer_source_review'],Path(r['roles']['pending_source_review']),sha(r['roles']['pending_source_review']))
assert l['exact_wsl_arguments']==arguments('clock')
for name,text,preview in zip(('run.ps1','run_durable.ps1'),texts(),('preview_run.ps1.txt','preview_durable.ps1.txt')):
    path=BASE/'clock_process'/name
    assert path.read_text(encoding='utf-8-sig')==text==(BASE/preview).read_text(encoding='utf-8-sig')
    assert pins[path.as_posix()]==sha(path)
assert read(BASE/'template_parse.json')['passed'] is True
for name in ('clock_process/launch_clearance.json','clock_process/started.lock','clock_process/start.json','clock_process/child.json','clock_process/exit.json','clock_process/dispatch.json','run','stage_receipts'):assert not (BASE/name).exists(),name
result=dict(passed=True,reviewer='root',reviewed_utc=datetime.now(timezone.utc).isoformat(),request_subject=dict(path=rp.as_posix(),sha256=sha(rp)),launch_receipt_subject=dict(path=lp.as_posix(),sha256=sha(lp)),checked_input_pins=len(pins),preserved_original_external_pins=external,all_current_input_pins_exact=True,unchanged_scope_fields=list(unchanged),exact_launch_arguments=True,actual_launchers_equal_parsed_previews=True,original_roles_unchanged=True,source_and_saved_auditor_reviews_bound=True,helper_review_subject=dict(path=helper_path.as_posix(),sha256=sha(helper_path)),producer_source_count=24,auditor_source_count=20,job_retry_max=10,result_retry_max=20,original_deadline_unchanged=True,dispatch_performed=False,execution_selected=False,root_selection_must_wait_for_no_active_policy_or_native_work=True,native_steps=0,model_calls=0,MJB_serializations=0,writer_sha256=sha(__file__),hardware_authorized=False)
with (OUT/'concrete_review.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(dict(passed=True,sha256=sha(OUT/'concrete_review.json'),pins=len(pins),external=external)))
