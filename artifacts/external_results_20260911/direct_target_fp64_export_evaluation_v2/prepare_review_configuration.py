"""Validate actual release audit/receipts before freezing a runtime binding.

No model, export, physics or training execution. This preserves the reviewed v2
freezer/runtime and binds this extra preflight into its immutable configuration.
"""
import argparse,hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent;NEW=BASE.parent;EXPORT=NEW/'direct_target_fp64_export_v1';FIT=NEW/'direct_target_continuation_v1'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def subject(path):
    path=Path(path).resolve();assert path.is_absolute() and path.is_file()
    return dict(path=path.as_posix(),sha256=sha(path))
def require_export_audit(audit,expected):
    assert audit['passed'] is True and audit['evidence_audit_passed'] is True and audit['export_qualified'] is True
    assert audit['canonical_evaluation_cleared'] is False
    direct=audit['direct_subject_sha256']
    for name,digest in expected.items():assert direct[name]==digest,('root_export_audit',name)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root-export-audit',type=Path,required=True)
    parser.add_argument('--export-review',type=Path,required=True);parser.add_argument('--export-pass-field',default='export_review_pass')
    a=parser.parse_args()
    destination=BASE/'release_reviews.json';assert not destination.exists()
    paths=dict(head=EXPORT/'export/student_head_fp64.onnx',checkpoint=FIT/'fit/student_head.pt',fit_report=FIT/'fit/report.json',
        source_head=FIT/'fit/student_head.onnx',normalization=FIT/'fit/normalization.npz',export_report=EXPORT/'export/report.json',
        export_request=EXPORT/'export_request.json',export_manifest=EXPORT/'export_frozen_inputs.json')
    subjects={name:subject(path) for name,path in paths.items()}
    expected={name:entry['sha256'] for name,entry in subjects.items()}
    root_training=subject(NEW/'direct_target_continuation_failure_audit_v1/results_v1/report.json')
    expected['root_training_audit']=root_training['sha256']
    root_export=subject(a.root_export_audit);require_export_audit(read(a.root_export_audit),expected)
    owner=subject(EXPORT/'owner_completion_verification.json');owner_record=read(owner['path'])
    assert owner_record['owner_verification_passed'] is True
    assert owner_record['raw_exit_known'] is True and owner_record['raw_python_exit_code']==owner_record['exit_code']==0
    assert owner_record['all_postrun_pins_exact'] is True and owner_record['processes_absent'] is True
    for name,entry in subjects.items():assert owner_record['direct_subject_sha256'][name]==entry['sha256']
    final_review=subject(a.export_review);review=read(a.export_review);assert review[a.export_pass_field] is True
    for name,digest in dict(expected,root_export_audit=root_export['sha256'],export_owner_completion=owner['sha256']).items():
        assert review['direct_subject_sha256'][name]==digest,('final_export_review',name)
    training=subject(NEW/'direct_target_continuation_training_only_review_v1/review.json')
    training_review=read(training['path']);assert training_review['training_review_pass'] is True and training_review['dataset_review_pass'] is True
    source=subject(NEW/'direct_target_fp64_export_evaluation_source_review_v1/review.json');assert read(source['path'])['source_review_pass'] is True
    def role(entry,pass_field):return dict(entry,pass_field=pass_field)
    result=dict(root_selected=True,selected_witness_calls=1,selected_main_controls=1569,conditional_hold_controls=250,
        root_training_audit=role(root_training,'evidence_audit_passed'),export_owner_completion=role(owner,'owner_verification_passed'),
        root_export_audit=role(root_export,'passed'),
        reviews=dict(dataset=role(training,'dataset_review_pass'),training=role(training,'training_review_pass'),
            export=role(final_review,a.export_pass_field),source=role(source,'source_review_pass')),
        prebinding_root_export_audit_passed=True,preflight_source=subject(__file__),
        actual_subjects=subjects,model_calls=0,native_steps=0,optimizer_updates=0,original_failed_export_preserved=True)
    with destination.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(configuration_sha256=sha(destination),actual_root_export_audit=root_export['sha256'],model_calls=0,native_steps=0)))
if __name__=='__main__':main()
