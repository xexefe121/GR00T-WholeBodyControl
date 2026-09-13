"""Record a concrete endpoint selection after completed paired-fit saved audit."""
import argparse,json,sys
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
EVAL=HERE.parent/'direct_target_causal_context_evaluation_v2'
FIT=HERE.parent/'direct_target_causal_context_study_v2'
AUDIT=HERE.parent/'direct_target_context_pair_fit_independent_v1'
sys.path.insert(0,str(EVAL))
from freeze_final_package import actual_paths,item,read,sha
from context_release import SUBJECTS,condition_report
from export_release_gate import direct_subjects,consumed_subjects

def main(a):
    assert not (EVAL/'root_final_release_review.json').exists()
    pair=FIT/'fit/paired_report.json';audit=AUDIT/'results_v1/report.json'
    owner=FIT/(a.condition+'_owner_completion_verification.json')
    assert sha(pair)==a.paired_report_sha256 and sha(audit)==a.audit_sha256 and sha(owner)==a.owner_sha256
    pair_value=read(pair);assert pair_value['completed'] is True
    audited=read(audit);assert audited['evidence_audit_passed'] is True and audited['export_qualified'] is True
    audit_owner=read(AUDIT/'owner_completion.json')
    assert audit_owner['completion_accounting_passed'] is True and audit_owner['evidence_audit_passed'] is True
    assert audit_owner['report_sha256']==sha(audit) and audit_owner['processes_absent'] is True
    paths=actual_paths(a.condition);subjects={role:item(paths[role]) for role in SUBJECTS}
    for condition in ('blinded','causal'):
        report=read(paths[condition+'_fit_report']);condition_report(report,condition)
        assert report['training_first_layer_execution']=='split_contiguous_1000_plus_323'
        assert report['export_first_layer_execution']=='monolithic_float64_1323'
        assert pair_value['condition_report_sha256'][condition]==sha(paths[condition+'_fit_report'])
    digests={role:entry['sha256'] for role,entry in subjects.items()}
    consumed_subjects(audited['input_sha256'],{entry['path']:entry['sha256'] for entry in subjects.values()})
    owned=read(owner);assert owned['owner_verification_passed'] is True
    assert owned['processes_absent'] is True and owned['raw_python_exit_code']==owned['exit_code']==0
    direct_subjects(owned,digests)
    review=HERE.parent/'direct_target_context_namespace_review_v2/review.json'
    assert sha(review)=='949b5124b0f452a917c87660a79f294e5276a8b20bc3da4c2a516d6d88dd3a29'
    source=read(review);assert source['passed'] is True
    for name,digest in source['source_sha256'].items():assert sha(EVAL/'source_draft_v1'/name)==digest
    for name,digest in source['helper_sha256'].items():assert sha(EVAL/name)==digest
    digests.update(root_training_audit=sha(audit),fit_owner_completion=sha(owner))
    result=dict(release_review_pass=True,reviewer='root',reviewed_utc=datetime.now(timezone.utc).isoformat(),
        context_condition=a.condition,ordinary_final_step=68000,direct_subject_sha256=digests,subjects=subjects,
        audit_path=str(audit),audit_owner_sha256=sha(AUDIT/'owner_completion.json'),
        source_and_helper_review_sha256=sha(review),paired_report_sha256=sha(pair),
        root_selection_reason=a.reason,same_source_weights_FP64_export=True,
        physical_qualification=False,selected_witness_calls=1,selected_main_controls=1569,conditional_hold_controls=250,
        per_stage_concrete_review_required=True,native_steps=0,model_calls=0,
        scope='One original canonical simulation pipeline; full same-controller acceptance remains pending.',writer_sha256=sha(__file__))
    out=EVAL/'root_final_release_review.json'
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'release_review_pass':True,'sha256':sha(out),'condition':a.condition}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--condition',choices=['blinded','causal'],required=True)
    for name in ('paired-report-sha256','audit-sha256','owner-sha256','reason'):p.add_argument('--'+name,required=True)
    main(p.parse_args())
