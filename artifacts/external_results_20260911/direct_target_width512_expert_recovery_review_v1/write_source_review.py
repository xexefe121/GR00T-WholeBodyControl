"""Bind completed source reading and synthetic checks. No task-array loading."""
import ast
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

BASE=Path(__file__).resolve().parent
NEW=BASE.parent
PRODUCER=NEW/'direct_target_width512_expert_recovery_v1'
PREPARATION=PRODUCER/'source_preparation_v2.json'
EXPECTED='1f7211ba3c08e985daf191b2c106682b97e6336db67b5a805e98d6b604f4b573'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def subject(p):return dict(path=Path(p).as_posix(),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def verify(s):assert sha(s['path'])==s['sha256'],s['path']


def main():
    assert sha(PREPARATION)==EXPECTED
    receipt=read(PREPARATION);source=Path(receipt['source_directory'])
    assert receipt['source_preparation_pass'] and receipt['source_only']
    assert not receipt['actual_saved_input_preparation'] and not receipt['actual_recovery_selected']
    actual={p.relative_to(source).as_posix():sha(p) for p in source.rglob('*.py')}
    assert actual==receipt['source_sha256'] and len(actual)==20
    for name in actual:ast.parse((source/name).read_text())
    previous=read(receipt['previous_preparation']['path']);verify(receipt['previous_preparation'])
    previous_source=Path(previous['source_directory'])
    assert {name:sha(previous_source/name) for name in previous['source_sha256']}==previous['source_sha256']
    changed=[name for name in actual if actual[name]!=previous['source_sha256'][name]]
    assert changed==['run_width251_actual_oracle.py']
    expected=(previous_source/changed[0]).read_text().replace(
        "save_work('first_query_complete')","save_work('initial_seed_certification_complete')")
    line="                        incoming_history=s['snapshot']['history_flat'].copy(),proposal_before_preview=np.asarray(True))"
    assert expected.count(line)==1
    expected=expected.replace(line,line+"\n                    save_work('first_optimized_target_ready')")
    assert expected==(source/changed[0]).read_text()
    qualified=NEW/'bfm_entry250_actual_oracle_v1'
    old=read(qualified/'frozen_inputs.json')
    old_map={k:v for k,v in old['source_sha256'].items() if k!='run_bfm250_actual_oracle.py'}
    assert len(old_map)==14 and old_map==receipt['unchanged_qualified_source_sha256']
    for name,digest in old_map.items():
        assert actual[name]==digest==sha(qualified/'source_snapshot_v1'/name)
    for item in receipt['artifacts'].values():verify(item)
    verify(receipt['synthetic_tests']['receipt'])
    verify(receipt['additional_tests']['receipt']);verify(receipt['additional_tests']['source'])
    verify(receipt['revision_source'])
    # Small existing qualification receipts are checked directly. Original trace
    # identity remains carried from the completed independent semantics audit;
    # the actual extraction and concrete launch must rehash that NPZ.
    for role,item in receipt['subjects'].items():
        if role!='trace':verify(item)
    tests=BASE/'final_synthetic_tests.xml'
    suites=ET.parse(tests).getroot().findall('testsuite')
    assert sum(int(s.attrib['tests']) for s in suites)==36
    assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ('errors','failures','skipped'))
    assert receipt['protocol']['requested_branch_controls']==1318
    assert receipt['protocol']['conditional_hold_controls']==250
    assert receipt['protocol']['prefix_native_steps']==2510
    b=receipt['protocol']['budgets']
    categories=['native_'+s+'_step' for s in ('BFM','initial_certificate','restoration_certificate','preview')]
    for field in ('attempted_calls_max','attempted_units_max'):
        assert sum(b[k][field] for k in categories)==b['native_private_step'][field]==200180
    assert b['native_live_step']['attempted_units_max']==15680
    report=dict(passed=True,source_review_pass=True,source_only=True,
        source_preparation_subject=subject(PREPARATION),source_preparation_sha256=EXPECTED,
        source_directory=source.as_posix(),source_sha256=actual,
        unchanged_qualified_source_sha256=old_map,
        protocol=receipt['protocol'],input_subjects=receipt['subjects'],
        tests=dict(passed=True,total=36,independent_new_cases=9,producer_cases_independently_rerun=27,
                   receipt=subject(tests),earlier_independent_synthetic_cases=10,
                   final_phase_delta_checked_as_exact_source_substitution=True),
        checks=dict(full291_boundary_and_warning_preservation=True,
            prefix251_controls_and2510_steps_not_new_work=True,
            incoming_prior_inverse_of_actual_learned250_target=True,
            all251_measured_history_entries_reconstructed=True,
            real_history_commits_once_after_preview=True,
            private_BFM_and_planning_never_mutate_actual_memory=True,
            original_source_frames_and_remaining1318_scope=True,
            hold250_only_after_complete_main_and_same_state_clock=True,
            attempted_and_returned_work_separate=True,
            four_private_categories_and_aggregate_capped_before_native_call=True,
            partial_batch_progress_unknown_preserved=True,
            initial_seed_and_optimized_target_distinguished=True,
            literal_future_snapshot_prefix_selection_roles_checked=True,
            unchanged_qualified_teacher_math14_modules=True),
        resolved_finding='Private native aggregate now has disjoint seed, initial-certificate, restoration-certificate and imminent-preview sub-budgets.',
        limitations=[
            'This receipt qualifies source and synthetic behavior, not an actual recovered rollout or concrete launch.',
            'Actual extraction and launch must rehash the full prespecified trace and bind all consumed native/model/runtime assets.',
            'Expected rejected BFM proposals can leave high-level attempted greater than returned; outcome and primitive evidence determine completion.',
            'Counters are checkpointed, not durably persisted before every call; a killed process without final task_ended evidence has incomplete accounting.',
            'A failed multistep Batch call exposes attempted lane-steps; completed internal lane progress remains unknown.',
            'The original expert bound tolerance is 1e-6 rad. The prior direct rollout used exact bounds; neither contract changes here.',
            'Initial H30 seed feasibility and first target proposal are not full-source/hold qualification or admissible labels. Prefix0..250 is excluded from expert labels.'
        ],
        task_array_loads=0,task_model_calls=0,native_calls=0,replans=0,optimizer_calls=0,
        actual_recovery_execution_cleared=False,labels_admissible=False,
        review_artifacts={p.name:subject(p) for p in [Path(__file__),BASE/'test_private_categories_final.py',BASE/'test_completion_contract.py']})
    out=BASE/'source_review.json'
    with out.open('x',newline='\n') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(review=subject(out),source_review_pass=True,sources=len(actual),tests=36)))


if __name__=='__main__':main()
