"""Pure metadata freeze of unchanged recorded inputs and reviewed timeout source.

Writes one fresh request only; no native/imported controller objects or worker.
The inherited command-table digest is bound by the exact original trace files.
"""
import copy
import json
import sys
from pathlib import Path

BASE=Path(__file__).resolve().parent;NEW=BASE.parent
OLD=NEW/'independent_plant_process_clock_v1';SOURCE=BASE/'source_draft_v1'
sys.path.insert(0,str(SOURCE))
from packet_io import read,write,sha,require_roles
from stage_watchdog import BUDGETS,validate_request

ORIGINAL_REQUEST='d6a6c3332039046c77ff9c93eaedd0694f0ac3c8487a474760c25646e75ba91a'
ORIGINAL_LAUNCH='4ecd884ea49e2b307febb63967c222946c9ef6fdce73a8b52c59571b9a383c80'
ROOT_REVIEW='01939a4bf82d4f1b2e19e7240ea0f4aa4e6eba49c7dac71116a1c24b509d13c2'
AUDIT_PREP='0a013cf050d4d68599d59b6f6031a275063eab55f6b4a2b8c56018f3363438fd'
AUDIT_REVIEW='278c509418ef1c7a1be655693a840217dd8c1894a6a6c33f0757a712ad8b9ac7'


def main():
    for name in ('clock_request.json','clock_process','run','stage_receipts'):
        if (BASE/name).exists():raise ValueError('Fresh source metadata only; preserve existing '+name)
    original_path=OLD/'clock_request.json';launch_path=OLD/'clock_process/launch_receipt.json'
    review_path=NEW/'independent_plant_clock_timeout_root_source_review_v1/review.json'
    audit_prep=NEW/'independent_plant_clock_saved_root_review_v1/source_preparation_audit_v3.json'
    audit_review=NEW/'independent_plant_clock_saved_root_review_v1/root_source_review_v3.json'
    for path,expected in ((original_path,ORIGINAL_REQUEST),(launch_path,ORIGINAL_LAUNCH),(review_path,ROOT_REVIEW),(audit_prep,AUDIT_PREP),(audit_review,AUDIT_REVIEW)):
        if sha(path)!=expected:raise ValueError('Actual preserved subject changed: '+str(path))
    original=read(original_path);launch=read(launch_path);review=read(review_path);prep=read(BASE/'source_preparation_v1.json')
    if review['passed'] is not True or review['source_sha256']!=prep['source_sha256']:raise ValueError('Exact reviewed timeout source required')
    audit_gate=read(audit_review)
    if audit_gate['passed'] is not True or audit_gate['source_sha256']!=read(audit_prep)['source_sha256']:raise ValueError('Exact independent saved-auditor review required')
    if sha(BASE/'source_preparation_v1.json')!=review['source_preparation_subject']['sha256']:raise ValueError('Reviewed source preparation differs')
    pins={};kept=[];excluded=[]
    def add(path,expected=None):
        path=Path(path).resolve();actual=sha(path)
        if expected is not None and actual!=expected:raise ValueError('Pinned file changed: '+str(path))
        key=path.as_posix()
        if key in pins and pins[key]['sha256']!=actual:raise ValueError('Conflicting input identity')
        pins[key]={'path':key,'sha256':actual};return actual
    # Retain every original external runtime/reference/native/witness input. Only
    # the historical task's code/docs/launch files are replaced by this version.
    if len(launch['input_hashes'])!=3718:raise ValueError('Unexpected original launch scope')
    for path,digest in launch['input_hashes'].items():
        p=Path(path).resolve()
        if p.is_relative_to(OLD.resolve()):excluded.append({'path':p.as_posix(),'sha256':digest})
        else:add(p,digest);kept.append(p.as_posix())
    if any(Path(p).resolve().is_relative_to(OLD.resolve()) for p in original['roles'].values()):raise ValueError('An original runtime role would be excluded')
    for name,digest in prep['source_sha256'].items():add(BASE/name,digest)
    for path in (Path(__file__),BASE/'source_preparation_v1.json',BASE/'source_changes.diff',BASE/'request_delta_proposal.json',
        BASE/'DESIGN.md',BASE/'stub_tests_v2.xml',BASE/'template_parse.json',review_path,audit_prep,audit_review,original_path,launch_path,
        OLD/'TIMEOUT_ASSESSMENT_V1.md',OLD/'timeout_preservation_v2/report.json'):
        add(path)
    for name,digest in read(audit_prep)['source_sha256'].items():add(audit_prep.parent/'source_audit_v3'/name,digest)
    request=copy.deepcopy(original)
    request.update(run_id='query250-recorded-clock-timeout-v2',input_epoch=2,source_directory=SOURCE.as_posix(),
        output_directory=(BASE/'run').as_posix(),stage_directory=(BASE/'stage_receipts').as_posix(),
        watchdog_budgets=dict(BUDGETS),outer_process_timeout_seconds=555,
        execution_selected=False,preparation_only=True,execution_requires_separate_root_clearance=True)
    request['byte_preserved_copies']={name:{'original_path':(OLD/'source_runner_v1'/name).as_posix(),'sha256':sha(SOURCE/name)}
        for name in prep['unchanged_runner_files']}
    for name,value in request['byte_preserved_copies'].items():
        if sha(value['original_path'])!=value['sha256']:raise ValueError('Preserved runner source differs: '+name)
    request['roles'].update(timeout_source_review=review_path.as_posix(),saved_stage_audit_preparation=audit_prep.as_posix(),saved_stage_audit_source_review=audit_review.as_posix())
    request['input_files']=[pins[k] for k in sorted(pins)]
    request['prior_failed_attempt']={'request_sha256':ORIGINAL_REQUEST,'launch_receipt_sha256':ORIGINAL_LAUNCH,
        'preservation_report_sha256':sha(OLD/'timeout_preservation_v2/report.json'),'repeat_not_selected':True}
    validate_request(request);require_roles(request)
    destination=BASE/'clock_request.json';write(destination,request)
    write(BASE/'input_scope_derivation.json',{'request_sha256':sha(destination),'original_request_sha256':ORIGINAL_REQUEST,
        'original_launch_sha256':ORIGINAL_LAUNCH,'original_launch_pin_count':3718,'kept_external_paths':kept,
        'excluded_historical_task_files':excluded,'new_input_pin_count':len(pins),'all_external_original_pins_exact':True,
        'original_roles_unchanged':all(request['roles'][k]==v for k,v in original['roles'].items()),
        'original_command_table_sha256':original['command_table_sha256'],'command_table_recomputed':False,
        'original_native_and_reference_bytes_unchanged':True,'new_MJB_serializations':0,'new_native_steps':0,
        'new_model_calls':0,'worker_processes_started':0,'execution_selected':False})
    print(json.dumps({'request_sha256':sha(destination),'input_pins':len(pins),'external_original_pins':len(kept),'execution_selected':False}))


if __name__=='__main__':main()
