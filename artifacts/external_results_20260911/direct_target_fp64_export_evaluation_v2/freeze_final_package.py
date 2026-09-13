"""Bind a separate positive FP64 release without changing the failed original fit."""
import argparse,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;SOURCE=BASE/'source_draft_v1'
FIT=BASE.parent/'direct_target_continuation_v1';EXPORT=BASE.parent/'direct_target_fp64_export_v1'
sys.path.insert(0,str(SOURCE))
from evaluation_gate import sha,read,field,has_hash,bound_file,require_model_ready,require_ready,CENTERS_SHA,LABEL_SHA
from export_release_gate import validate_release

def item(path):
    p=Path(path).resolve();assert p.is_absolute() and p.is_file(),str(p)
    return dict(path=p.as_posix(),sha256=sha(p))
def add(pins,path,digest=None):
    entry=item(path)
    if digest is not None:assert entry['sha256']==digest,str(path)
    assert entry['path'] not in pins or pins[entry['path']]==entry['sha256']
    pins[entry['path']]=entry['sha256'];return entry
def role(spec):
    entry=item(spec['path']);entry['pass_field']=spec['pass_field']
    assert field(read(entry['path']),entry['pass_field']) is True
    return entry
def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['witness','evaluation'],required=True)
    p.add_argument('--reviews',type=Path,required=True);a=p.parse_args();config=read(a.reviews)
    assert config['root_selected'] is True
    assert config['selected_witness_calls']==1 and config['selected_main_controls']==1569 and config['conditional_hold_controls']==250
    assert not (BASE/(a.mode+'_binding.json')).exists()
    for name in ('head_witness',) if a.mode=='witness' else ('nominal','post_lifecycle_hold_5s','pilot_outcome.json'):
        assert not (BASE/name).exists(),name
    inventory=read(BASE/'runtime_inventory.json');derivation=read(BASE/'source_derivation.json')
    assert inventory['recursive_training_hashes'] is False and derivation['passed'] is True
    current={p.relative_to(SOURCE).as_posix():sha(p) for p in SOURCE.rglob('*.py')}
    assert current=={name:entry['sha256'] for name,entry in derivation['sources'].items()}
    pins={}
    for entry in inventory['files']:add(pins,entry['path'],entry['sha256'])
    paths=dict(head=EXPORT/'export/student_head_fp64.onnx',export_report=EXPORT/'export/report.json',
        export_request=EXPORT/'export_request.json',export_manifest=EXPORT/'export_frozen_inputs.json',
        checkpoint=FIT/'fit/student_head.pt',source_head=FIT/'fit/student_head.onnx',fit_report=FIT/'fit/report.json',
        normalization=FIT/'fit/normalization.npz',training_manifest=FIT/'training_frozen_inputs.json',training_request=FIT/'training_request.json',
        centers=BASE.parent/'velocity_chord_student_v1/generation/centers.npz',
        query250_labels=BASE.parent/'bfm_entry250_labels_v1/labels/labels.npz',
        contract=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json'))
    final={name:add(pins,path) for name,path in paths.items()}
    assert final['centers']['sha256']==CENTERS_SHA and final['query250_labels']['sha256']==LABEL_SHA
    root=role(config['root_training_audit']);owner=role(config['export_owner_completion'])
    reviews={name:role(config['reviews'][name]) for name in ('dataset','training','export','source')}
    for entry in [root,owner,*reviews.values()]:add(pins,entry['path'],entry['sha256'])
    for path in (Path(__file__),BASE/'prepare_bound_launcher.py',BASE/'diagnostic_verdict.py',BASE/'verify_completed_stage.py',
        BASE/'test_launch_helpers.py',BASE/'launch_helper_tests_v2.xml',BASE/'source_stub_tests.xml',BASE/'release_gate_tests.xml',
        BASE/'runtime_inventory.json',BASE/'source_derivation.json',BASE/'source_preparation.json',a.reviews):add(pins,path)
    b=dict(**final,reviews=reviews,root_training_audit=root,export_owner_completion=owner,
        onnx_dependencies=inventory['onnx_dependencies'],training_dataset_sha256=[final['training_manifest']['sha256']],
        dataset_subject_semantics='completed training manifest qualified by immutable positive training-only review',
        recursive_training_hashes=False,runtime_inventory_sha256=sha(BASE/'runtime_inventory.json'),
        ordinary_final_step=55000,controller='direct_absolute_target_1000_applied_inverse_learned_prior',
        release_kind='same55000_checkpoint_fp64_export',export_input_dtype='float32',export_output_dtype='float32',export_internal_dtype='float64',
        architecture=[1000,256,256,23],hidden_activation='ELU',head_output='normalized_target',
        span_contract='existing_float32_joint_span_promoted_float64',learned_BFM_calls=0,hardware_authorized=False,
        clock_foundation_connected=False,filters_enabled=False,compiled_preview_enabled=False,
        root_authorized_witness=True,root_authorized_evaluation=True,expected_head_calls=1,
        physics_authorized=a.mode=='evaluation',requested_main_controls=1569,conditional_hold_controls=250)
    # Check actual positive subject-bound release BEFORE creating a binding.
    validate_release(b,paths,SOURCE,a.mode,read=read,sha=sha,bound_file=bound_file,field=field,has_hash=has_hash)
    if a.mode=='evaluation':
        b['first_export_witness']=add(pins,BASE/'head_witness/witness.npz')
        b['first_export_receipt']=add(pins,BASE/'head_witness/report.json')
        for name in ('witness_binding.json','witness_completion_verification.json','witness_process/exit.json',
            'witness_process/raw_exit.json','witness_process/diagnostic_verdict.json','witness_process/postrun_hashes.json',
            'witness_process/launch_receipt.json','witness_process/launch_clearance.json'):
            add(pins,BASE/name)
        assert read(BASE/'witness_completion_verification.json')['diagnostic_passed'] is True
        assert read(BASE/'witness_process/exit.json')['exit_code']==0
        assert read(BASE/'witness_process/raw_exit.json')['raw_python_exit_code']==0
        frozen=BASE/'frozen_inputs_v2.json'
        write_new(frozen,dict(kind='one_same55000_fp64_export_canonical_evaluation',source_sha256=current,input_sha256=pins,
            requested_main_controls=1569,conditional_hold_controls=250,learned_BFM_calls=0,
            recursive_training_hashes=False,original_failed_fit_preserved=True,hardware_authorized=False))
        add(pins,frozen)
    b['input_files']=[dict(path=p,sha256=d) for p,d in sorted(pins.items())]
    path=BASE/(a.mode+'_binding.json');write_new(path,b)
    require_model_ready(BASE,'witness') if a.mode=='witness' else require_ready(BASE)
    print(json.dumps(dict(binding_sha256=sha(path),pins=len(pins),recursive_training_hashes=False,model_calls=0,native_steps=0)))
if __name__=='__main__':main()
