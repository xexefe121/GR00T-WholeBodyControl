"""Root check of prepared BUSY retry clock package. Does not select execution."""
from pathlib import Path
import hashlib,json,sys,unittest,io
OUT=Path(__file__).resolve().parent;NEW=OUT.parent
BASE=NEW/'independent_plant_pending_publication_v1';PRIOR=NEW/'independent_plant_clock_timeout_correction_v1';OLD=NEW/'independent_plant_process_clock_v1'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
request_path=BASE/'clock_request.json';receipt_path=BASE/'clock_process/launch_receipt.json'
assert sha(request_path)=='2b538b03eb25de6d4c2ea3165561371af2415089957f4b9dd0bf9020a893118c'
assert sha(receipt_path)=='c7cc875c5f4c1d36e9efef5a56c8091d638e66d48393a0086bb2b30681c4dc3e'
request=read(request_path);receipt=read(receipt_path);prep=read(BASE/'helper_preparation.json')
for name,h in prep['helper_sha256'].items():assert sha(BASE/name)==h
for name,h in prep['evidence_sha256'].items():assert sha(BASE/name)==h
for name,h in prep['unchanged_helper_sha256'].items():assert sha(PRIOR/name)==h
assert len(prep['unchanged_helper_sha256'])==3
for name,h in prep['producer_source_sha256'].items():assert sha(BASE/'source_draft_v1'/name)==h
audit_source=NEW/'independent_plant_pending_publication_saved_audit_v1/source_draft_v2'
for name,h in prep['auditor_source_sha256'].items():assert sha(audit_source/name)==h
assert len(prep['producer_source_sha256'])==22 and len(prep['auditor_source_sha256'])==17
sys.path.insert(0,str(BASE));import prepare_clock_stage as stage
from prepare_concrete_packet import validate_retry_contract,RETRY_DETAILS
from stage_watchdog import validate_request
validate_request(request);validate_retry_contract(request)
assert receipt['exact_wsl_arguments']==stage.arguments('clock')
assert receipt['requested_native_steps']==18190 and receipt['requested_serializations']==4
assert receipt['execution_selected'] is False and receipt['final_clearance_required'] is True
assert receipt['automatic_retry'] is receipt['hardware_authorized'] is False
assert request['input_epoch']==3 and request['execution_selected'] is False
for name,text in zip(('run.ps1','run_durable.ps1'),stage.texts()):assert (BASE/'clock_process'/name).read_text(encoding='utf-8-sig')==text
pins=receipt['input_hashes'];assert len(pins)==3768
for path,h in pins.items():assert sha(path)==h,path
request_pins={p['path']:p['sha256'] for p in request['input_files']}
for path,h in request_pins.items():assert pins[path]==h
old=read(OLD/'clock_process/launch_receipt.json');original=read(OLD/'clock_request.json')
external={path:h for path,h in old['input_hashes'].items() if not Path(path).resolve().is_relative_to(OLD.resolve())}
assert len(external)==3689
for path,h in external.items():assert request_pins[Path(path).resolve().as_posix()]==h
assert all(request['roles'][name]==path for name,path in original['roles'].items())
assert request['command_table_sha256']==original['command_table_sha256']
assert not (BASE/'clock_process/launch_clearance.json').exists()
assert not (BASE/'clock_process/started.lock').exists() and not (BASE/'run').exists() and not (BASE/'stage_receipts').exists()
import test_launch_helpers
log=io.StringIO();result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_launch_helpers))
(OUT/'root_tests.log').write_text(log.getvalue(),encoding='utf-8')
assert result.wasSuccessful() and result.testsRun==12 and not result.skipped
report=dict(passed=True,source_review_pass=True,helper_review_pass=True,concrete_review_pass=True,reviewer='root',request_subject=dict(path=request_path.as_posix(),sha256=sha(request_path)),launch_receipt_subject=dict(path=receipt_path.as_posix(),sha256=sha(receipt_path)),helper_preparation_sha256=sha(BASE/'helper_preparation.json'),helper_sha256=prep['helper_sha256'],source_sha256=prep['producer_source_sha256'],retry_auditor_source_sha256=prep['auditor_source_sha256'],root_helper_tests=12,all_current_pins_exact=True,checked_input_pins=3768,retained_original_external_pins=3689,exact_launch_arguments=True,requested_native_steps=18190,requested_serializations=4,original_controls_and_deadlines_unchanged=True,reviewed_semantics=['Same immutable job on BUSY, max10attempts/once per eligible tick/original activation; original command table and all external runtime inputs preserved.','Source and separate retry-aware saved auditor reviews bind exact22+17maps before request creation.','Original durable hidden captured-handle/CreateNew/knownrawexit/posthash wrapper and owner/verdict unchanged.','Exact new request and launcher require separate final clearance. No epoch or deadline rebase; no hardware.'],execution_selected=False,dispatch_performed=False,idle_compute_slot_required=True,physical_qualification=False,timing_qualification=False,model_calls=0,native_steps=0,writer_sha256=sha(__file__))
with (OUT/'review.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'passed':True,'sha256':sha(OUT/'review.json'),'pins':3768,'tests':12,'execution_selected':False}))
