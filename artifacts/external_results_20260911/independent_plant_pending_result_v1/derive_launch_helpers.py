"""Add helper sources only; do not freeze requests or start any process."""
from pathlib import Path
import difflib,hashlib,json

BASE=Path(__file__).resolve().parent
PRIOR=BASE.parent/'independent_plant_pending_publication_v1'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    copies={}
    for name in ('prepare_clock_stage.py','prepare_stage_preserved_template.py','stage_verdict.py','verify_completion.py','check_templates.ps1'):
        with (BASE/name).open('xb') as f:f.write((PRIOR/name).read_bytes())
        copies[name]={'original_path':(PRIOR/name).as_posix(),'sha256':sha(BASE/name)}
    text=(PRIOR/'prepare_concrete_packet.py').read_text()
    replacements=[
        ('one future recorded-command BUSY retry benchmark','one future recorded-command job/result BUSY retry benchmark'),
        ("PRIOR=NEW/'independent_plant_clock_timeout_correction_v1'", "PRIOR=NEW/'independent_plant_pending_publication_v1'\nTIMEOUT=NEW/'independent_plant_clock_timeout_correction_v1'"),
        ("AUDIT=NEW/'independent_plant_pending_publication_saved_audit_v1'", "AUDIT=NEW/'independent_plant_pending_result_saved_audit_v1'"),
        ('6cb9f8bf38543634c32e745496f63fe94e23391fd10948be41749eaea5356575','88080d6c84025b7589a381a1d5d3dc5f170acdb69b20c95f31123f3c6278b4ad'),
        ('d2f303a488817a1f088790fbe37a8d411576ddfb7b84c1043c5309dbc7a4e357','b017dbb753fe2c8a0370288de01eb9e52275e2e5a365b54582da20cb15700640'),
        ('e656ed762475a50f91cf60b4bfc3b59ce9c052ae44fad54c5474e41060adcfde','09bf03da9bae5285ccc24cabd77a83718e06485dbc267a50d047deceea1f3d91'),
        ('81c4090dd6035fc4b39064fd98e7412af87e2d0355a4a3abd260d19ffb4faae0','009d21d7dcc8b347da5d0f511ae2d2c24c22774eef84f8ba12f1e7101cb37a48'),
        ('c67d970962a1fe67258154d13beb82c04a69f70c25426e77736def8c790770d1','76c54a589689411b1f4dd7c4937b9485f7979ec21065408c5cb51cfb67443626'),
        ("ambiguous_publication_retry=False,worker_result_retry=False,", "ambiguous_publication_retry=False,worker_result_retry=True,"),
        ("HELPERS=('prepare_concrete_packet.py'", "RESULT_CONTRACT='immutable_result_BUSY_max20_original_deadline'\nDEADLINE_CONTRACT='plant_epoch_plus_activation_20ms'\nRESULT_DETAILS=dict(max_attempts_per_result=20,max_attempts_per_worker_iteration=1,\n    worker_iteration_sleep_seconds=0.001,max_owned_polled_jobs=2,retry_status='BUSY',\n    same_immutable_result_and_job=True,literal_original_job_deadline=True,\n    pending_blocks_new_job_poll=True,recompute_reply=False,\n    late_publication_does_not_override_admission=True,\n    ambiguous_publication_retry=False,blocking_retry=False,deadline_rebase=False)\nHELPERS=('prepare_concrete_packet.py'"),
        ("    details=request.get('job_publication_retry_details')\n", "    if request.get('worker_result_retry_contract')!=RESULT_CONTRACT or request.get('job_deadline_contract')!=DEADLINE_CONTRACT:\n        raise ValueError('Exact reviewed worker result and literal deadline contracts required')\n    validate_details(request.get('worker_result_retry_details'),RESULT_DETAILS)\n    validate_details(request.get('job_publication_retry_details'),RETRY_DETAILS)\n\n\ndef validate_details(details,expected):\n"),
        ('set(details)!=set(RETRY_DETAILS)', 'set(details)!=set(expected)'),
        ('for key,value in RETRY_DETAILS.items():','for key,value in expected.items():'),
        ("def check_review(prep,review,prep_sha,expected_count):", "def check_subject(subject,path,digest):\n    if type(subject) is not dict or set(subject)!={'path','sha256'}:\n        raise ValueError('Dedicated subject path and SHA required')\n    if type(subject['path']) is not str or Path(subject['path']).resolve()!=Path(path).resolve() or subject['sha256']!=digest:\n        raise ValueError('Dedicated subject differs')\n\n\ndef check_review(prep,review,prep_path,prep_sha,expected_count,subject_field):"),
        ("    if review.get('source_preparation_sha256')!=prep_sha or review.get('source_sha256')!=prep.get('source_sha256'):", "    check_subject(review.get(subject_field),prep_path,prep_sha)\n    if review.get('source_sha256')!=prep.get('source_sha256'):"),
        ("NEW/'independent_pending_publication_root_review_v1/review.json'", "NEW/'independent_pending_result_source_review_v1/review.json'"),
        ("NEW/'independent_pending_publication_saved_root_review_v1/review.json'", "NEW/'independent_pending_result_saved_audit_review_v1/review.json'"),
        ("check_review(prep,read(review_path),SOURCE_PREP,22)", "check_review(prep,read(review_path),prep_path,SOURCE_PREP,24,'source_preparation_subject')"),
        ("check_review(aprep,read(audit_review),AUDIT_PREP,17)", "check_review(aprep,read(audit_review),audit_prep,AUDIT_PREP,20,'preparation_subject')"),
        ("    if audit_gate['pending_core_sha256']!=CORE_SHA or audit_gate['required_clock_request_contract']!=RETRY_CONTRACT:\n        raise ValueError('Auditor does not cover this producer contract')", "    check_subject(audit_gate.get('producer_preparation'),prep_path,SOURCE_PREP)\n    check_subject(audit_gate.get('producer_source_review'),review_path,ROOT_REVIEW)"),
        ("BASE/'clock_core_final.diff'", "BASE/'source_changes.diff'"),
        ("NEW/'independent_clock_timing_diagnosis_v1/report.json')", "NEW/'independent_clock_timing_diagnosis_v1/report.json',\n        TIMEOUT/'clock_request.json',TIMEOUT/'run/report.json',TIMEOUT/'owner_completion.json',\n        NEW/'independent_plant_pending_publication_saved_actual_v1/results_v1/report.json',\n        NEW/'independent_plant_pending_publication_saved_actual_v1/owner_completion_dispatch_v3.json',\n        NEW/'independent_pending_clock_publication_diagnosis_v1/report.json',\n        NEW/'independent_pending_clock_publication_diagnosis_v1/diagnostic_receipt.json',BASE/'OUTPUT_SCHEMA_DELTA.md')"),
        ("query250-recorded-clock-pending-BUSY-v1", "query250-recorded-clock-pending-result-BUSY-v1"),
        ("job_publication_retry_contract=RETRY_CONTRACT,job_publication_retry_details=dict(RETRY_DETAILS),", "job_publication_retry_contract=RETRY_CONTRACT,job_publication_retry_details=dict(RETRY_DETAILS),\n        worker_result_retry_contract=RESULT_CONTRACT,worker_result_retry_details=dict(RESULT_DETAILS),\n        job_deadline_contract=DEADLINE_CONTRACT,"),
        ("prep['unchanged_original_files']", "prep['unchanged_files']"),
        ("retry_contract=RETRY_CONTRACT,new_MJB_serializations=0", "retry_contract=RETRY_CONTRACT,worker_result_retry_contract=RESULT_CONTRACT,job_deadline_contract=DEADLINE_CONTRACT,new_MJB_serializations=0"),
    ]
    for old,new in replacements:
        if old not in text:raise ValueError('Missing exact derivation fragment: '+old)
        text=text.replace(old,new)
    start=text.index('    for path in (prep_path,')
    end=text.index('        add(path)\n',start)+len('        add(path)\n')
    expression=text[start+len('    for path in '):end-len(':\n        add(path)\n')]
    signature='prep_path,review_path,audit_prep,audit_review,original_path,launch_path'
    function='def antecedent_paths('+signature+'):\n    return '+expression+'\n\n\n'
    text=text[:start]+'    for path in antecedent_paths('+signature+'):\n        add(path)\n'+text[end:]
    text=text.replace('def main():\n',function+'def main():\n',1)
    with (BASE/'prepare_concrete_packet.py').open('x',encoding='utf-8') as f:f.write(text)
    diff=''.join(difflib.unified_diff((PRIOR/'prepare_concrete_packet.py').read_text().splitlines(True),text.splitlines(True),fromfile='pending_publication/prepare_concrete_packet.py',tofile='pending_result/prepare_concrete_packet.py'))
    with (BASE/'helper_derivation.diff').open('x',encoding='utf-8') as f:f.write(diff)
    with (BASE/'helper_derivation.json').open('x') as f:json.dump({'unchanged_copies':copies,'source_only':True,'actual_dispatch':False,'requests_created':0,'launchers_created':0},f,indent=2)

if __name__=='__main__':main()
