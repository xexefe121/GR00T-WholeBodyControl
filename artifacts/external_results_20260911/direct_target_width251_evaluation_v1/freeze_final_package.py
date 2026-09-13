"""Future explicitly selected width512 recovery91000 release binding. Requires actual positive receipts first."""
import argparse,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;SOURCE=BASE/'source_draft_v1';NEW=BASE.parent
FIT=NEW/'direct_target_width251_student_v1'
sys.path.insert(0,str(SOURCE))
from evaluation_gate import sha,read,field,bound_file,has_hash,require_model_ready,require_ready,CENTERS_SHA,LABEL_SHA
from context_release import validate_release,SUBJECTS


def item(path):
    path=Path(path).resolve();assert path.is_absolute() and path.is_file(),str(path)
    return {'path':path.as_posix(),'sha256':sha(path)}


def add(pins,path,digest=None):
    entry=item(path)
    if digest is not None:assert entry['sha256']==digest
    assert entry['path'] not in pins or pins[entry['path']]==entry['sha256']
    pins[entry['path']]=entry['sha256'];return entry


def role(spec):
    entry=item(spec['path']);entry['pass_field']=spec['pass_field']
    if 'sha256' in spec:assert entry['sha256']==spec['sha256']
    assert field(read(entry['path']),entry['pass_field']) is True
    return entry


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def require_helper_review(spec):
    entry=role(spec);review=read(entry['path']);prep=read(BASE/'launch_helper_preparation_v2.json')
    expected=prep['helper_sha256']
    assert expected and review['helper_sha256']==expected
    for name,digest in expected.items():assert sha(BASE/name)==digest,name
    return entry


def require_condition(condition):
    if type(condition) is not str or condition!='causal':
        raise ValueError('Only the explicitly selected causal continuation condition is allowed.')
    return condition


def actual_paths(condition):
    condition=require_condition(condition)
    request=read(FIT/'training_request.json');dest=FIT/'fit';shared=FIT/'fit/shared'
    return dict(head=dest/'student_head.onnx',checkpoint=dest/'student_head.pt',fit_report=dest/'report.json',
        normalization=shared/'normalization.npz',training_manifest=FIT/'training_frozen_inputs.json',training_request=FIT/'training_request.json',
        export_manifest=dest/'output_manifest.json',coefficient=Path(request['subjects']['coefficient_source']['path']),
        source_checkpoint=Path(request['subjects']['checkpoint']['path']),
        full_state_generation_request=Path(request['full_state_paths']['request']),
        full_state_generation_report=Path(request['full_state_paths']['report']),
        full_state_data_audit=Path(request['subjects']['full_state_root_audit']['path']),
        full_state_data_owner=Path(request['subjects']['full_state_root_owner']['path']),
        shared_manifest=shared/'output_manifest.json',context_alignment=shared/'context_alignment.json',
        energy_source=Path(request['subjects']['energy_source']['path']),
        **{role:Path(request['subjects'][role]['path']) for role in ('recovery_rows','collection_report','collection_request','collection_qualification','collection_source_review','consistency_report','warm_restore_review')},
        centers=NEW/'velocity_chord_student_v1/generation/centers.npz',query250_labels=NEW/'bfm_entry250_labels_v1/labels/labels.npz',
        contract=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json'))


def base_binding(config,final,inventory):
    assert config['root_selected'] is True
    condition=require_condition(config['context_condition'])
    assert config['selected_witness_calls']==1 and config['selected_main_controls']==1569 and config['conditional_hold_controls']==250
    return dict(**final,reviews={name:role(config['reviews'][name]) for name in ('release','source')},
        root_training_audit=role(config['root_training_audit']),fit_owner_completion=role(config['fit_owner_completion']),
        onnx_dependencies=inventory['onnx_dependencies'],ordinary_final_step=91000,context_condition=condition,
        release_kind='ordinary91000_width512_recovery_context_same_weight_fp64_export',export_input_dtype='float32',export_output_dtype='float32',export_internal_dtype='float64',
        controller='direct_absolute_target_1323_causal_width512_recovery',architecture=[1323,512,512,23],hidden_activation='ELU',head_output='normalized_target',
        span_contract='existing_float32_joint_span_promoted_float64',learned_BFM_calls=0,hardware_authorized=False,
        clock_foundation_connected=False,filters_enabled=False,compiled_preview_enabled=False,
        root_authorized_witness=True,root_authorized_evaluation=True,expected_head_calls=1,
        requested_main_controls=1569,conditional_hold_controls=250,recursive_training_hashes=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['witness','evaluation'],required=True)
    parser.add_argument('--reviews',type=Path,required=True);args=parser.parse_args();config=read(args.reviews)
    assert not (BASE/(args.mode+'_binding.json')).exists()
    for name in ('head_witness',) if args.mode=='witness' else ('nominal','post_lifecycle_hold_5s','pilot_outcome.json'):
        assert not (BASE/name).exists(),name
    inventory=read(BASE/'runtime_inventory.json');prep=read(BASE/'source_preparation.json');pins={}
    assert inventory['recursive_training_hashes'] is False and inventory['baseline_full291_fixture_byteexact'] is True
    current={p.relative_to(SOURCE).as_posix():sha(p) for p in SOURCE.rglob('*.py')}
    assert current==prep['source_sha256']==inventory['source_sha256']
    paths=actual_paths(config['context_condition']);final={name:add(pins,path) for name,path in paths.items()}
    assert final['centers']['sha256']==CENTERS_SHA and final['query250_labels']['sha256']==LABEL_SHA
    b=base_binding(config,final,inventory);b['physics_authorized']=args.mode=='evaluation'
    # Check actual release before any binding write or task inference.
    validate_release(b,paths,SOURCE,args.mode,read=read,sha=sha,bound_file=bound_file,field=field,has_hash=has_hash)
    assert config['subject_sha256']=={name:b[name]['sha256'] for name in SUBJECTS}
    helper_review=require_helper_review(config['launch_helper_review'])
    for entry in inventory['files']:add(pins,entry['path'],entry['sha256'])
    for entry in [b['root_training_audit'],b['fit_owner_completion'],helper_review,*b['reviews'].values()]:add(pins,entry['path'],entry['sha256'])
    for name in ('freeze_final_package.py','prepare_bound_launcher.py','diagnostic_verdict.py','verify_completed_stage.py',
                 'prepare_review_configuration.py','test_launch_helpers.py','test_release_helpers.py','launch_helper_tests_v3.xml',
                 'runtime_inventory.json','source_preparation.json','launch_helper_preparation_v2.json','powershell_path_normalization_v1.json'):
        add(pins,BASE/name)
    add(pins,args.reviews)
    if args.mode=='evaluation':
        b['first_export_witness']=add(pins,BASE/'head_witness/witness.npz')
        b['first_export_receipt']=add(pins,BASE/'head_witness/report.json')
        for name in ('witness_binding.json','witness_completion_verification.json','witness_process/exit.json','witness_process/raw_exit.json',
                     'witness_process/diagnostic_verdict.json','witness_process/postrun_hashes.json','witness_process/launch_receipt.json','witness_process/launch_clearance.json'):
            add(pins,BASE/name)
        witness_binding=read(BASE/'witness_binding.json')
        assert witness_binding['context_condition']==b['context_condition'] and witness_binding['head']==b['head']
        owner=read(BASE/'witness_completion_verification.json')
        assert owner['owner_completion_accounting_passed'] is True and owner['diagnostic_passed'] is True
        assert read(BASE/'witness_process/exit.json')['exit_code']==0 and read(BASE/'witness_process/raw_exit.json')['raw_python_exit_code']==0
        frozen=BASE/'frozen_inputs_v1.json';write_new(frozen,dict(kind='one_ordinary91000_width512_recovery_canonical_evaluation',context_condition=b['context_condition'],
            source_sha256=current,input_sha256=pins,requested_main_controls=1569,conditional_hold_controls=250,
            recursive_training_hashes=False,learned_BFM_calls=0,hardware_authorized=False));add(pins,frozen)
    b['input_files']=[{'path':path,'sha256':digest} for path,digest in sorted(pins.items())]
    path=BASE/(args.mode+'_binding.json');write_new(path,b)
    require_model_ready(BASE,'witness') if args.mode=='witness' else require_ready(BASE)
    print(json.dumps({'binding_sha256':sha(path),'pins':len(pins),'native_steps':0,'head_calls':0}))


if __name__=='__main__':main()
