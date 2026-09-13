"""Consolidate completed producer and independent saved-data audit evidence.

No array math, model, feature, teacher-map, optimizer or native calls.
"""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

BASE=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')
OUT=Path(__file__).resolve().parent
GEN=BASE/'direct_target_full_state_secants_v1'
AUD=BASE/'direct_target_full_state_data_root_review_v1'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def win(path):
    text=str(path).replace('\\','/')
    return Path(text[5].upper()+':/'+text[7:]) if text.startswith('/mnt/') else Path(text)
def pinmap(value):return {win(p).as_posix():digest for p,digest in value.items()}

expected={
 'generation_request':(GEN/'request.json','474d5e4a04f84702b8ada696560df4baf6a3ebfa3b0ffebb0c343d4da11ebde1'),
 'generation_report':(GEN/'generation/report.json','68c9cd0ac47d59e2bb6d14269769b98f60ea6aa3a03298f82e4d164e140f4144'),
 'generation_manifest':(GEN/'generation/manifest.json','2c337d78ca7c9d291ab0e07f95fa34a3175c9e245aef4ff79303b6e57f3a4638'),
 'generation_owner':(GEN/'owner_completion_verification.json','444728bd9b1e3e895976bec8ba1ac9a8ccf7b1189d2d9e8a8fba80cab2d3d619'),
 'root_audit':(AUD/'results_v1/report.json','6eae25bc3d212de13b0940dae11d85d1739aa1d26ce17c549b2b6d5a36f55564'),
 'root_owner':(AUD/'owner_completion_v2.json','3f47a899d3e08ebb9d5a1ec514e395563dd677c4209a281f774f76401b401e43'),
 'conflicts':(AUD/'results_v1/duplicate_conflicts.json','e560497c96d5b577b10574dc9387e63053abf61be6bfa6decdd6e50d03a32327'),
 'root_source_review':(BASE/'direct_target_full_state_auditor_pico_review_v2/review.json','a6ed5dc0cdd757fb06792539435feea4ff54b233c8d9ba47c9b05451c82e42e2')}
subjects={}
for name,(path,digest) in expected.items():
    assert sha(path)==digest,(name,'changed')
    subjects[name]=dict(path=path.as_posix(),sha256=digest)
producer=read(expected['generation_report'][0]);owner=read(expected['generation_owner'][0])
manifest=read(expected['generation_manifest'][0]);audit=read(expected['root_audit'][0])
audit_owner=read(expected['root_owner'][0]);conflicts=read(expected['conflicts'][0])
source_review=read(expected['root_source_review'][0])
assert producer['complete'] is True and producer['all_slots_valid'] is True
assert producer['original_center_and_overlap_exact'] is True and owner['passed'] is True
assert producer['request_sha256']==owner['request_sha256']==audit['request_sha256']==expected['generation_request'][1]
assert producer['manifest_sha256']==owner['manifest_sha256']==expected['generation_manifest'][1]
assert owner['report_sha256']==audit['generation_report_sha256']==expected['generation_report'][1]
assert audit['data_review_pass'] is True and audit_owner['passed'] is True and audit_owner['data_review_pass'] is True
assert audit['checks']==1241497 and audit['rows_checked']==354612 and audit['full_tangent_and_state_rows']==354612
assert audit['exact_old_overlap_rows']==140622 and audit['independently_recomputed_new_features_and_maps']==213990
assert audit['corpus_sizes']==conflicts['corpus_sizes']==dict(nominal=9904,full_state=354612,physical=3054)
assert audit['unique_numerical_feature_rows']==conflicts['unique_numerical_feature_rows']==367570
assert audit['duplicate_rows']==conflicts['duplicate_rows']==0
assert audit['incompatible_normalized_float32_target_rows']==conflicts['incompatible_normalized_float32_target_rows']==0
assert conflicts['duplicates']==conflicts['conflicts']==[]
assert source_review['source_review_passed'] is True and source_review['actual_saved_audit_source_clear'] is True
assert audit_owner['processes_absent'] is True and audit_owner['raw_exit_known'] is True
assert audit_owner['raw_python_exit_code']==audit_owner['exit_code']==0
assert audit_owner['all_current_input_pins_exact'] is True and audit_owner['input_pin_count']==50
assert audit_owner['saved_data_audit_repeated'] is False
assert audit_owner['invalid_prior_receipt']['valid'] is False
audit_pins=pinmap(audit['input_sha256'])
for key in ('generation_request','generation_report','generation_manifest'):
    assert audit_pins[subjects[key]['path']]==subjects[key]['sha256']
for path,digest in source_review['input_sha256'].items():
    if win(path).name in ('audit_saved_data.py','saved_math.py') and 'source_preserved_v1' not in str(path):
        assert audit_pins[win(path).as_posix()]==digest and sha(win(path))==digest
current_metadata={}
for path,digest in audit_owner['output_sha256'].items():
    assert sha(win(path))==digest
    current_metadata[win(path).as_posix()]=digest
start=read(AUD/'process_v1/start.json');child=read(AUD/'process_v1/child.json');exit_record=read(AUD/'process_v1/exit.json');absence=read(AUD/'process_absence.json')
assert start['wrapper_pid']==child['wrapper_pid']==exit_record['wrapper_pid']==absence['wrapper_pid']==audit_owner['wrapper_pid']==20280
assert child['child_pid']==exit_record['child_pid']==absence['child_pid']==audit_owner['child_pid']==27728
assert child['handle_captured'] is True and absence['processes_absent'] is True
assert exit_record['raw_python_exit_code']==exit_record['exit_code']==0 and exit_record['error'] is None and exit_record['all_postrun_pins_exact'] is True
for path,digest in start['pins'].items():
    assert sha(win(path))==digest
    current_metadata[win(path).as_posix()]=digest
assert start['producer_report_sha256']==expected['generation_report'][1]
assert audit['duplicate_report_sha256']==expected['conflicts'][1]
for key in ('model_calls','BFM_calls','native_steps','optimizer_updates','replans'):assert audit[key]==0
for cell,count in enumerate((100,819,100)*3):
    assert audit['complete54cell_counts'][cell]==[count*size*2 for size in (3,3,23,3,3,23)]
for name,entry in manifest['arrays'].items():
    path=(GEN/'generation'/entry['path']).as_posix()
    assert audit_pins[path]==entry['sha256']
    subjects['full_state_'+name]=dict(path=path,sha256=entry['sha256'])
for name,relative in [('full_state_centers','direct_target_full_state_secants_v1/generation/centers.npz'),
                      ('original_centers','velocity_chord_student_v1/generation/centers.npz'),
                      ('pico','pico_walk002_labels_v1/collection/pico/labels.npz'),
                      ('walk002','pico_walk002_labels_v1/collection/walk002/labels.npz'),
                      ('physical_manifest','one_step_policy_branch_collection_resume2969_v1/collection/data/manifest.json')]:
    path=(BASE/relative).as_posix();subjects[name]=dict(path=path,sha256=audit_pins[path])
review=dict(passed=True,data_review_pass=True,dataset_review_pass=True,created_utc=datetime.now(timezone.utc).isoformat(),
    generation_request_sha256=expected['generation_request'][1],generation_report_sha256=expected['generation_report'][1],
    generation_manifest_sha256=expected['generation_manifest'][1],root_audit_sha256=expected['root_audit'][1],root_owner_sha256=expected['root_owner'][1],
    subjects=subjects,direct_subject_sha256={name:subject['sha256'] for name,subject in subjects.items()},
    independently_audited_rows=354612,exact_old_overlap_rows=140622,added_rows=213990,nominal_rows=9904,physical_rows=3054,
    full54cells_complete=True,unique_training_inputs=367570,duplicate_inputs=0,conflicting_normalized_float32_labels=0,rows_removed=0,
    audit_source_review_passed=True,root_owner_v2_valid=True,invalid_owner_v1_preserved=True,
    producer_owner_preceded_independent_audit=True,producer_owner_pending_field_superseded_only_by_this_separate_completed_review=True,
    review_scope='Consolidated completed-data qualification from producer and independently executed root saved-data audit; this reviewer produced the corpus and did not repeat the independent all-array computation.',
    verification_scope='Current metadata/source digests and exact subject/map/manifest/exit/PID linkage checked here; full array hashes and semantics rely on completed root audit and corrected owner v2.',
    current_metadata_sha256=current_metadata,audited_input_sha256=audit_pins,
    model_calls=0,feature_calls=0,map_calls=0,native_steps=0,optimizer_updates=0,
    training_launch_cleared=False,export_qualified=False,canonical_cleared=False,hardware_authorized=False,
    limitations=[audit['limitation'],'Static admissibility and exact committed maps do not establish contact feasibility or closed-loop recovery.'],
    review_source_sha256=sha(__file__))
output=OUT/'review.json'
with output.open('x',encoding='utf-8') as f:json.dump(review,f,indent=2,allow_nan=False);f.write('\n')
print(output.as_posix(),sha(output))
