"""Bind the completed recovery fit to the existing full-motion simulator."""
import json,sys
from pathlib import Path
from datetime import datetime,timezone
BASE=Path(__file__).resolve().parent;NEW=BASE.parent
sys.path.insert(0,str(BASE/'source_draft_v1'))
from evaluation_gate import read,sha
from context_release import SUBJECTS
from freeze_final_package import actual_paths,item,write_new
from prepare_review_configuration import configuration

audit_base=NEW/'direct_target_width251_fit_independent_v2'
audit_path=audit_base/'results_v1/report.json';audit_owner_path=audit_base/'owner_completion.json'
audit=read(audit_path);audit_owner=read(audit_owner_path)
assert audit['evidence_audit_passed'] is audit['export_qualified'] is True
for key in ('completion_accounting_passed','evidence_audit_passed','export_qualified','processes_absent','raw_exit_known','all_pre_post_current_launch_pins_exact'):
    assert audit_owner[key] is True,key
assert audit_owner['raw_python_exit_code']==audit_owner['exit_code']==0
assert audit_owner['report_sha256']==sha(audit_path)
paths=actual_paths('causal');subjects={name:item(paths[name]) for name in SUBJECTS}
fit=read(paths['fit_report']);fit_owner_path=NEW/'direct_target_width251_student_v1/owner_completion_verification.json'
fit_owner=read(fit_owner_path)
assert fit['completed'] is True and fit['ordinary_final_step']==91000 and fit['optimizer_step']==26000
assert fit_owner['owner_verification_passed'] is fit_owner['completion_passed'] is True
for name,entry in subjects.items():
    assert audit['direct_subject_sha256'][name]==fit_owner['direct_subject_sha256'][name]==entry['sha256'],name
direct={name:entry['sha256'] for name,entry in subjects.items()}
direct.update(root_training_audit=sha(audit_path),fit_owner_completion=sha(fit_owner_path))
release_path=BASE/'root_final_release_review.json'
write_new(release_path,dict(release_review_pass=True,reviewer='root',reviewed_utc=datetime.now(timezone.utc).isoformat(),
    direct_subject_sha256=direct,subjects=subjects,ordinary_final_step=91000,
    selected_witness_calls=1,selected_main_controls=1569,conditional_hold_controls=250,
    root_selection_reason='Test fast controller on full original walk003 and uninterrupted standing hold. Training loss is not the acceptance result.',
    audit_owner_sha256=sha(audit_owner_path),model_calls=0,native_steps=0,physical_qualification=False))
source=BASE/'source_root_review.json'
def role(path,field):return {'path':str(path),'pass_field':field}
config=configuration(root_selected=True,condition='causal',release=role(release_path,'release_review_pass'),
    root_audit=role(audit_path,'evidence_audit_passed'),fit_owner=role(fit_owner_path,'owner_verification_passed'),
    source_review=role(source,'passed'),helper_review=role(source,'passed'))
write_new(BASE/'release_reviews.json',config)
print(json.dumps({'ready_for_full_trial':True,'head_sha256':subjects['head']['sha256'],'release_sha256':sha(release_path)}))
