"""Source-only helper port. No request, launcher, clearance or task execution."""
from pathlib import Path
import hashlib,json
BASE=Path(__file__).resolve().parent;NEW=BASE.parent
PRIOR=NEW/'independent_plant_pending_result_v1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(path,text):
    with path.open('x',encoding='utf-8',newline='') as f:f.write(text)
def replace(text,old,new):
    assert text.count(old)==1,(old,text.count(old));return text.replace(old,new)

def main():
    old=BASE/'helper_original_v1';old.mkdir(exist_ok=False)
    names=('prepare_concrete_packet.py','prepare_clock_stage.py','prepare_stage_preserved_template.py',
        'stage_verdict.py','verify_completion.py','test_launch_helpers.py','record_helper_preparation.py','check_templates.ps1')
    for name in names:
        with (old/name).open('xb') as f:f.write((PRIOR/name).read_bytes())
    for name in ('prepare_clock_stage.py','prepare_stage_preserved_template.py','check_templates.ps1'):
        with (BASE/name).open('xb') as f:f.write((old/name).read_bytes())
    text=(old/'prepare_concrete_packet.py').read_text()
    text=replace(text,'import copy,json,sys','import argparse,copy,json,sys')
    text=replace(text,"PRIOR=NEW/'independent_plant_pending_publication_v1'","PRIOR=NEW/'independent_plant_pending_result_v1'")
    text=replace(text,"AUDIT=NEW/'independent_plant_pending_result_saved_audit_v1'","AUDIT=NEW/'independent_plant_timing_saved_audit_v1'")
    text=replace(text,"SOURCE_PREP='88080d6c84025b7589a381a1d5d3dc5f170acdb69b20c95f31123f3c6278b4ad'","SOURCE_PREP='3844f7d86a4f97557c8a74744f7ea382a25315ae3a96cb3abfa786992d2c8cc5'")
    text=replace(text,"ROOT_REVIEW='b017dbb753fe2c8a0370288de01eb9e52275e2e5a365b54582da20cb15700640'","ROOT_REVIEW='fdccbf0e31e716190a805cd2d5e6f52d3aae629e18136386f4da9d1e95b78310'")
    text=replace(text,"AUDIT_PREP='09bf03da9bae5285ccc24cabd77a83718e06485dbc267a50d047deceea1f3d91'","AUDIT_PREP='78d7855373d409fcd6c3a44edef433d6be520faaaf956a739483dad2b8dd6ae8'")
    text=replace(text,"AUDIT_REVIEW='009d21d7dcc8b347da5d0f511ae2d2c24c22774eef84f8ba12f1e7101cb37a48'\n",'')
    text=replace(text,"CORE_SHA='76c54a589689411b1f4dd7c4937b9485f7979ec21065408c5cb51cfb67443626'","CORE_SHA='e0fc679a8cc23a7d18e500bbf37dd2d5e82fa90e89a5692bfd0953dbce5534ff'\nTIMING_CONTRACT='preallocated_wall_thread_process_GC_v2'")
    text=replace(text,"    'derive_launch_helpers.py')","    'derive_launch_helpers.py','sidecar_owner_evidence.py','test_sidecar_owner.py')")
    text=replace(text,"def validate_retry_contract(request):\n","def validate_retry_contract(request):\n    if request.get('timing_probe_contract')!=TIMING_CONTRACT:\n        raise ValueError('Exact reviewed timing sidecar contract required')\n")
    text=replace(text,'    check_subject(review.get(subject_field),prep_path,prep_sha)',"    if subject_field=='source_preparation_sha256':\n        if review.get(subject_field)!=prep_sha:raise ValueError('Literal preparation digest differs')\n    else:check_subject(review.get(subject_field),prep_path,prep_sha)")
    text=replace(text,"BASE/'source_changes.diff',BASE/'helper_derivation.json',BASE/'helper_derivation.diff',BASE/'PACKET_SCOPE.md'","BASE/'integration_final.diff',BASE/'helper_derivation.json',BASE/'helper_derivation.diff',BASE/'PACKET_SCOPE.md'")
    text=replace(text,"BASE/'OUTPUT_SCHEMA_DELTA.md')","BASE/'OUTPUT_SCHEMA_DELTA.md',\n        NEW/'independent_plant_pending_result_saved_actual_v1/results_v1/report.json',\n        NEW/'independent_plant_pending_result_saved_actual_v1/owner_completion.json',\n        NEW/'independent_pending_result_clock_diagnosis_v1/report.json',\n        NEW/'independent_pending_result_clock_diagnosis_v1/diagnostic_receipt.json',\n        NEW/'independent_timing_probe_root_review_v1/review.json')")
    text=replace(text,'def main():\n',"def main():\n    parser=argparse.ArgumentParser()\n    parser.add_argument('--audit-source-review',type=Path,required=True)\n    parser.add_argument('--audit-source-review-sha256',required=True)\n    args=parser.parse_args()\n    if len(args.audit_source_review_sha256)!=64 or any(c not in '0123456789abcdef' for c in args.audit_source_review_sha256):\n        raise ValueError('Actual auditor review SHA required')\n")
    text=replace(text,"review_path=NEW/'independent_pending_result_source_review_v1/review.json'","review_path=NEW/'independent_timing_integration_root_review_v1/review.json'")
    text=replace(text,"audit_review=NEW/'independent_pending_result_saved_audit_review_v1/review.json'","audit_review=args.audit_source_review.resolve()")
    text=replace(text,'(audit_review,AUDIT_REVIEW)', '(audit_review,args.audit_source_review_sha256)')
    text=replace(text,"check_review(prep,read(review_path),prep_path,SOURCE_PREP,24,'source_preparation_subject')","check_review(prep,read(review_path),prep_path,SOURCE_PREP,32,'source_preparation_sha256')")
    text=replace(text,"check_review(aprep,read(audit_review),audit_prep,AUDIT_PREP,20,'preparation_subject')","check_review(aprep,read(audit_review),audit_prep,AUDIT_PREP,25,'source_preparation_sha256')")
    text=replace(text,"    audit_gate=read(audit_review)\n    check_subject(audit_gate.get('producer_preparation'),prep_path,SOURCE_PREP)\n    check_subject(audit_gate.get('producer_source_review'),review_path,ROOT_REVIEW)\n",'')
    text=replace(text,"run_id='query250-recorded-clock-pending-result-BUSY-v1',input_epoch=3","run_id='query250-recorded-clock-timing-instrumentation-v1',input_epoch=4")
    text=replace(text,"        job_deadline_contract=DEADLINE_CONTRACT,\n","        job_deadline_contract=DEADLINE_CONTRACT,timing_probe_contract=TIMING_CONTRACT,\n")
    text=replace(text,"for name in prep['unchanged_files']}","for name in prep['unchanged_original_modules']}")
    text=replace(text,"'timing_failure_remains_independent_of_BUSY_fix':True","'timing_failure_remains_independent_of_BUSY_fix':True,\n        'instrumentation_is_diagnostic_not_optimization':True")
    put(BASE/'prepare_concrete_packet.py',text)
    text=(old/'stage_verdict.py').read_text()
    text=replace(text,'from packet_io import read,write,sha','from packet_io import read,write,sha\nfrom sidecar_owner_evidence import check_sidecars')
    text=replace(text,'passed=False;error=None;report=None','passed=False;error=None;report=None;sidecars=None')
    text=replace(text,"        stages=[read(p)","        sidecars=check_sidecars(output,report,read(output/'output_manifest.json'))\n        stages=[read(p)")
    text=replace(text,'preservation_complete and report',"preservation_complete and sidecars['instrumentation_complete'] is True and report")
    text=replace(text,"'root_saved_audit_pending':True,'error':error","'root_saved_audit_pending':True,'timing_sidecar_accounting':sidecars,'error':error")
    put(BASE/'stage_verdict.py',text)
    text=(old/'verify_completion.py').read_text()
    text=replace(text,'from packet_io import read,write,sha,pin_check','from packet_io import read,write,sha,pin_check\nfrom sidecar_owner_evidence import check_sidecars')
    text=replace(text,"    assert report['request_sha256']==manifest", "    sidecars=check_sidecars(output,report,manifest)\n    assert diagnostic['timing_sidecar_accounting']==sidecars\n    assert report['request_sha256']==manifest")
    text=replace(text,"'diagnostic_passed':diagnostic['diagnostic_passed'],'component_qualified':False", "'diagnostic_passed':diagnostic['diagnostic_passed'],'timing_sidecar_accounting':sidecars,'component_qualified':False")
    put(BASE/'verify_completion.py',text)
    text=(old/'test_launch_helpers.py').read_text()
    text=replace(text,'DEADLINE_CONTRACT,validate_retry_contract','DEADLINE_CONTRACT,TIMING_CONTRACT,validate_retry_contract')
    text=replace(text,'job_deadline_contract=DEADLINE_CONTRACT)','job_deadline_contract=DEADLINE_CONTRACT,timing_probe_contract=TIMING_CONTRACT)')
    text=replace(text,"'job_deadline_contract'):","'job_deadline_contract','timing_probe_contract'):")
    text=replace(text,"('source_preparation_subject','preparation_subject')", "('source_preparation_subject','preparation_subject','source_preparation_sha256')")
    text=replace(text,"good[field]={'path':path.as_posix(),'sha256':'b'*64}","good[field]='b'*64 if field=='source_preparation_sha256' else {'path':path.as_posix(),'sha256':'b'*64}")
    text=replace(text,'independent_plant_pending_result_v1/source_draft_v1','independent_plant_timing_integration_v1/source_draft_v1')
    put(BASE/'test_launch_helpers.py',text)
    record=dict(preparation_only=True,original_helper_sha256={name:sha(old/name) for name in names},
        unchanged_helper_sha256={name:sha(BASE/name) for name in ('prepare_clock_stage.py','prepare_stage_preserved_template.py','check_templates.ps1')},
        actual_request_created=False,actual_clearance_created=False,actual_dispatch=False)
    put(BASE/'helper_derivation.json',json.dumps(record,indent=2)+'\n')

if __name__=='__main__':main()
