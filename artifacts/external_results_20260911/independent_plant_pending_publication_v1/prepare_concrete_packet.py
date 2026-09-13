"""Freeze metadata for one future recorded-command BUSY retry benchmark.

No task-array decoding, model/native/worker calls or execution selection.
Original command-table digest and all external original inputs are inherited.
"""
import copy,json,sys
from pathlib import Path

BASE=Path(__file__).resolve().parent;NEW=BASE.parent
OLD=NEW/'independent_plant_process_clock_v1'
PRIOR=NEW/'independent_plant_clock_timeout_correction_v1'
SOURCE=BASE/'source_draft_v1'
AUDIT=NEW/'independent_plant_pending_publication_saved_audit_v1'
sys.path.insert(0,str(SOURCE))
from packet_io import read,write,sha,require_roles
from stage_watchdog import BUDGETS,validate_request

ORIGINAL_REQUEST='d6a6c3332039046c77ff9c93eaedd0694f0ac3c8487a474760c25646e75ba91a'
ORIGINAL_LAUNCH='4ecd884ea49e2b307febb63967c222946c9ef6fdce73a8b52c59571b9a383c80'
SOURCE_PREP='6cb9f8bf38543634c32e745496f63fe94e23391fd10948be41749eaea5356575'
ROOT_REVIEW='d2f303a488817a1f088790fbe37a8d411576ddfb7b84c1043c5309dbc7a4e357'
AUDIT_PREP='e656ed762475a50f91cf60b4bfc3b59ce9c052ae44fad54c5474e41060adcfde'
AUDIT_REVIEW='81c4090dd6035fc4b39064fd98e7412af87e2d0355a4a3abd260d19ffb4faae0'
CORE_SHA='c67d970962a1fe67258154d13beb82c04a69f70c25426e77736def8c790770d1'
RETRY_CONTRACT='pending_BUSY_same_job_max10_before_original_activation'
RETRY_DETAILS=dict(max_attempts_per_job=10,max_attempts_per_physics_tick=1,
    retry_status='BUSY',same_immutable_job_and_payload=True,
    original_activation_and_deadline=True,retry_guard_strictly_before_deadline=True,
    late_publication_does_not_override_admission=True,
    ambiguous_publication_retry=False,worker_result_retry=False,
    blocking_retry=False,deadline_rebase=False)
HELPERS=('prepare_concrete_packet.py','prepare_clock_stage.py','prepare_stage_preserved_template.py',
    'stage_verdict.py','verify_completion.py','test_launch_helpers.py','record_helper_preparation.py',
    'derive_launch_helpers.py')


def validate_retry_contract(request):
    if request.get('job_publication_retry_contract')!=RETRY_CONTRACT:
        raise ValueError('Exact reviewed pending BUSY contract required')
    details=request.get('job_publication_retry_details')
    if type(details) is not dict or set(details)!=set(RETRY_DETAILS):
        raise ValueError('Exact retry detail fields required')
    for key,value in RETRY_DETAILS.items():
        if type(details[key]) is not type(value) or details[key]!=value:
            raise ValueError('Retry scope changed: '+key)


def check_review(prep,review,prep_sha,expected_count):
    if review.get('passed') is not True or review.get('source_review_pass') is not True:
        raise ValueError('Positive literal source review required')
    if review.get('source_preparation_sha256')!=prep_sha or review.get('source_sha256')!=prep.get('source_sha256'):
        raise ValueError('Source review does not bind actual preparation and full map')
    if len(prep['source_sha256'])!=expected_count:raise ValueError('Unexpected reviewed source count')


def main():
    for name in ('clock_request.json','clock_process','run','stage_receipts','input_scope_derivation.json'):
        if (BASE/name).exists():raise ValueError('Preserve existing '+name)
    original_path=OLD/'clock_request.json';launch_path=OLD/'clock_process/launch_receipt.json'
    prep_path=BASE/'source_preparation.json'
    review_path=NEW/'independent_pending_publication_root_review_v1/review.json'
    audit_prep=AUDIT/'source_preparation_v2.json'
    audit_review=NEW/'independent_pending_publication_saved_root_review_v1/review.json'
    for path,expected in ((original_path,ORIGINAL_REQUEST),(launch_path,ORIGINAL_LAUNCH),
        (prep_path,SOURCE_PREP),(review_path,ROOT_REVIEW),(audit_prep,AUDIT_PREP),(audit_review,AUDIT_REVIEW)):
        if sha(path)!=expected:raise ValueError('Actual preserved subject changed: '+str(path))
    original=read(original_path);launch=read(launch_path);prep=read(prep_path);aprep=read(audit_prep)
    check_review(prep,read(review_path),SOURCE_PREP,22)
    check_review(aprep,read(audit_review),AUDIT_PREP,17)
    if prep['source_sha256']['clock_core.py']!=CORE_SHA:raise ValueError('Wrong pending core')
    audit_gate=read(audit_review)
    if audit_gate['pending_core_sha256']!=CORE_SHA or audit_gate['required_clock_request_contract']!=RETRY_CONTRACT:
        raise ValueError('Auditor does not cover this producer contract')
    helpers=read(BASE/'helper_preparation.json')
    if helpers.get('passed') is not True or set(helpers['helper_sha256'])!=set(HELPERS):
        raise ValueError('Complete helper preparation required')
    pins={};kept=[];excluded=[]
    def add(path,expected=None):
        path=Path(path).resolve();actual=sha(path)
        if expected is not None and actual!=expected:raise ValueError('Pinned file changed: '+str(path))
        key=path.as_posix()
        if key in pins and pins[key]['sha256']!=actual:raise ValueError('Conflicting input identity')
        pins[key]={'path':key,'sha256':actual};return actual
    if len(launch['input_hashes'])!=3718:raise ValueError('Unexpected original launch scope')
    for path,digest in launch['input_hashes'].items():
        p=Path(path).resolve()
        if p.is_relative_to(OLD.resolve()):excluded.append({'path':p.as_posix(),'sha256':digest})
        else:add(p,digest);kept.append(p.as_posix())
    if any(Path(p).resolve().is_relative_to(OLD.resolve()) for p in original['roles'].values()):
        raise ValueError('Original consumed role would be excluded')
    for name,digest in prep['source_sha256'].items():add(SOURCE/name,digest)
    for name,digest in aprep['source_sha256'].items():add(AUDIT/'source_draft_v2'/name,digest)
    for name,digest in helpers['helper_sha256'].items():add(BASE/name,digest)
    for name,digest in helpers['evidence_sha256'].items():add(BASE/name,digest)
    for path in (prep_path,review_path,audit_prep,audit_review,original_path,launch_path,BASE/'helper_preparation.json',
        BASE/'clock_core_final.diff',BASE/'helper_derivation.json',BASE/'helper_derivation.diff',BASE/'PACKET_SCOPE.md',
        PRIOR/'clock_request.json',PRIOR/'clock_process/launch_receipt.json',PRIOR/'run/report.json',PRIOR/'owner_completion.json',
        PRIOR/'source_draft_v1/clock_core.py',OLD/'TIMEOUT_ASSESSMENT_V1.md',OLD/'timeout_preservation_v2/report.json',
        NEW/'independent_plant_clock_saved_actual_v1/results_v1/report.json',NEW/'independent_plant_clock_saved_actual_v1/owner_completion.json',
        NEW/'independent_clock_publication_diagnosis_v1/diagnostic_receipt.json',
        NEW/'independent_clock_timing_diagnosis_v1/report.json'):
        add(path)
    request=copy.deepcopy(original)
    request.update(run_id='query250-recorded-clock-pending-BUSY-v1',input_epoch=3,source_directory=SOURCE.as_posix(),
        output_directory=(BASE/'run').as_posix(),stage_directory=(BASE/'stage_receipts').as_posix(),
        watchdog_budgets=dict(BUDGETS),outer_process_timeout_seconds=555,
        job_publication_retry_contract=RETRY_CONTRACT,job_publication_retry_details=dict(RETRY_DETAILS),
        execution_selected=False,preparation_only=True,execution_requires_separate_root_clearance=True)
    request['byte_preserved_copies']={name:{'original_path':(PRIOR/'source_draft_v1'/name).as_posix(),'sha256':sha(SOURCE/name)}
        for name in prep['unchanged_original_files']}
    for name,value in request['byte_preserved_copies'].items():
        if sha(value['original_path'])!=value['sha256']:raise ValueError('Preserved source changed: '+name)
    request['roles'].update(pending_source_review=review_path.as_posix(),
        saved_stage_audit_preparation=audit_prep.as_posix(),saved_stage_audit_source_review=audit_review.as_posix())
    request['input_files']=[pins[k] for k in sorted(pins)]
    request['prior_failed_attempt']={'request_sha256':sha(PRIOR/'clock_request.json'),
        'report_sha256':sha(PRIOR/'run/report.json'),'owner_sha256':sha(PRIOR/'owner_completion.json'),
        'original_timeout_request_sha256':ORIGINAL_REQUEST,'repeat_not_selected':True,
        'timing_failure_remains_independent_of_BUSY_fix':True}
    validate_request(request);validate_retry_contract(request);require_roles(request)
    destination=BASE/'clock_request.json';write(destination,request)
    write(BASE/'input_scope_derivation.json',dict(request_sha256=sha(destination),original_request_sha256=ORIGINAL_REQUEST,
        original_launch_sha256=ORIGINAL_LAUNCH,original_launch_pin_count=3718,kept_external_paths=kept,
        excluded_historical_task_files=excluded,new_input_pin_count=len(pins),all_external_original_pins_exact=True,
        original_roles_unchanged=all(request['roles'][k]==v for k,v in original['roles'].items()),
        original_command_table_sha256=original['command_table_sha256'],command_table_recomputed=False,
        original_native_and_reference_bytes_unchanged=True,changed_request_keys=sorted(k for k in request if request.get(k)!=original.get(k)),
        retry_contract=RETRY_CONTRACT,new_MJB_serializations=0,new_native_steps=0,new_model_calls=0,
        worker_processes_started=0,execution_selected=False))
    print(json.dumps(dict(request_sha256=sha(destination),input_pins=len(pins),external_original_pins=len(kept),execution_selected=False)))

if __name__=='__main__':main()
