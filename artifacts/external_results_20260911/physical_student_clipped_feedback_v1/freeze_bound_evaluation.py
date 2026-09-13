"""Bind the selected single feedback experiment to real unchanged final artifacts."""
from pathlib import Path
import argparse
import copy
import json
import sys

BASE=Path(__file__).parent;SOURCE=BASE/'source_snapshot_v1'
sys.path.insert(0,str(SOURCE))
from evaluation_gate import read,sha,bound_file,field,has_hash,require_ready
from feedback_parity import ORIGINAL_TRACE_SHA
ORIGINAL=BASE.parent/'one_step_physical_student_evaluation_v1'

def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')

def item(path):
    path=Path(path).resolve();assert path.is_file()
    return dict(path=path.as_posix(),sha256=sha(path))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-review',type=Path,required=True)
    parser.add_argument('--source-review-sha256',required=True);args=parser.parse_args()
    assert sha(args.source_review)==args.source_review_sha256
    review=read(args.source_review)
    assert review['source_review_pass'] is True
    assert has_hash(review,sha(SOURCE/'evaluate_physical_response_student.py'))
    for name in ('evaluation_binding.json','frozen_inputs_v2.json','evaluation_process','nominal','post_lifecycle_hold_5s','pilot_outcome.json','head_witness'):
        assert not (BASE/name).exists(),name
    original_binding=ORIGINAL/'evaluation_binding.json'
    assert sha(original_binding)=='eee281fc2d1d316367d31838140f5cbef25d44f62e965180354a55933647030c'
    binding=copy.deepcopy(read(original_binding))
    pins={entry['path']:entry['sha256'] for entry in binding['input_files']}
    for path,digest in pins.items():assert sha(path)==digest,path
    for name,digest in read(BASE/'source_freeze.json')['source_sha256'].items():
        assert sha(SOURCE/name)==digest,name;pins[(SOURCE/name).as_posix()]=digest
    original_trace=ORIGINAL/'nominal/trace.npz'
    assert sha(original_trace)==ORIGINAL_TRACE_SHA
    # These completed saved-data reports are lineage evidence, not new runtime calls.
    extras=[original_binding,original_trace,ORIGINAL/'nominal/report.json',ORIGINAL/'completion_verification.json',
        ORIGINAL/'witness_completion_verification.json',ORIGINAL/'evaluation_process/launch_receipt.json',ORIGINAL/'evaluation_process/exit.json',
        BASE.parent/'student_physical_response_independent_physics_v1/report.json',
        BASE.parent/'student_physical_response_saved_outcome_v1/report.json',
        BASE.parent/'student_physical_response_fixed_map_v1/report.json',
        args.source_review,Path(__file__),BASE/'prepare_bound_launcher.py',BASE/'SELECTION.md',BASE/'original_sources.json',
        BASE/'source_freeze.json',BASE/'source_derivation.patch',BASE/'freeze_sources.py',BASE/'test_clipped_feedback.py',
        BASE/'tests.json',BASE/'tests.log',BASE/'clipped_control264_example.json']
    for path in extras:pins[item(path)['path']]=sha(path)
    binding.update(controller='learned_native_clipped_component_feedback',
        unclipped_float32_feedback_preserved=True,initial_and_terminal_BFM_feedback_unchanged=True,
        reused_witness_only=True,additional_witness_calls=0,original_raw_trace=item(original_trace),
        requested_main_controls=1569,conditional_hold_controls=250,root_authorized_canonical_evaluation=True,
        first_changed_previous_action_control=265,first_changed_actor_lag_action_control=266,
        original_prefix_native_steps_required=2650,hardware_authorized=False)
    binding['reviews']['source']=dict(**item(args.source_review),pass_field='source_review_pass')
    frozen=BASE/'frozen_inputs_v2.json'
    write_new(frozen,dict(kind='one_selected_clipped_component_feedback_final75000',
        source_sha256=read(BASE/'source_freeze.json')['source_sha256'],input_sha256=pins,
        only_learned_clipped_components_change_feedback=True,raw_combined_action_logged_separately=True,
        original_BFM_startup_terminal_unchanged=True,head_and_physics_unchanged=True,reused_witness_no_new_call=True,
        hardware_authorized=False))
    pins[frozen.as_posix()]=sha(frozen)
    binding['input_files']=[dict(path=p,sha256=d) for p,d in sorted(pins.items())]
    write_new(BASE/'evaluation_binding.json',binding)
    require_ready(BASE)
    print(json.dumps(dict(binding_sha256=sha(BASE/'evaluation_binding.json'),frozen_sha256=sha(frozen),pins=len(pins),models_called=0,native_steps=0)))

if __name__=='__main__':main()
