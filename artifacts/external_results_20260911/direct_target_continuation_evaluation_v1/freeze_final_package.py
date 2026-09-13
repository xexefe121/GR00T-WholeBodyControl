"""Bind a completed reviewed final55000 export using actual runtime pins only.

Training/data authenticity belongs to immutable completed qualification receipts.
No traversal of their referenced training files occurs here or in runtime gates.
"""
import argparse,json
from pathlib import Path
import sys

BASE=Path(__file__).resolve().parent;SOURCE=BASE/'source_draft_v1'
FIT=BASE.parent/'direct_target_continuation_v1'
sys.path.insert(0,str(SOURCE))
from evaluation_gate import sha,read,field,has_hash,bound_file,require_model_ready,require_ready,CENTERS_SHA,LABEL_SHA

def item(path):
    path=Path(path).resolve();assert path.is_absolute() and path.is_file(),str(path)
    return dict(path=path.as_posix(),sha256=sha(path))
def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def role(spec,subjects):
    entry=item(spec['path']);entry['pass_field']=spec['pass_field'];r=read(bound_file(entry))
    assert field(r,entry['pass_field']) is True
    assert subjects and all(has_hash(r,d) for d in subjects)
    return entry
def add(pins,path,digest=None):
    e=item(path)
    if digest is not None:assert e['sha256']==digest,str(path)
    assert e['path'] not in pins or pins[e['path']]==e['sha256']
    pins[e['path']]=e['sha256'];return e

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
    assert current=={name:e['sha256'] for name,e in derivation['sources'].items()}
    pins={}
    for entry in inventory['files']:add(pins,entry['path'],entry['sha256'])
    final={name:add(pins,path) for name,path in dict(head=FIT/'fit/student_head.onnx',
        checkpoint=FIT/'fit/student_head.pt',fit_report=FIT/'fit/report.json',
        training_manifest=FIT/'training_frozen_inputs.json',training_request=FIT/'training_request.json',
        centers=BASE.parent/'velocity_chord_student_v1/generation/centers.npz',
        query250_labels=BASE.parent/'bfm_entry250_labels_v1/labels/labels.npz',
        contract=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json')).items()}
    fit=read(final['fit_report']['path'])
    assert fit['ordinary_final_step']==55000 and fit['additional_updates']==50000
    assert fit['features']==1000 and fit['head_output']=='normalized_target'
    for key in ('optimization_completed','final_export_diagnostics_completed','numerical_gate_passed','export_parity_passed'):
        assert fit[key] is True,key
    assert fit['onnx_sha256']==final['head']['sha256'] and fit['checkpoint_sha256']==final['checkpoint']['sha256']
    assert final['centers']['sha256']==CENTERS_SHA and final['query250_labels']['sha256']==LABEL_SHA
    # Final export/data reviewers directly qualify this immutable full training manifest.
    # Its nested hashes are evidence, not files traversed by the evaluator.
    subject=[final[k]['sha256'] for k in ('fit_report','head','checkpoint')]
    root=role(config['root_audit'],subject)
    owner=role(config['owner_completion'],subject)
    producer=add(pins,FIT/'fit_process/exit.json');producer_record=read(producer['path'])
    assert producer_record['raw_python_exit_code']==producer_record['exit_code']==0
    assert producer_record['exit_known'] is True and producer_record['error'] is None
    assert producer_record['all_postrun_pins_exact'] is True
    assert has_hash(read(owner['path']),producer['sha256'])
    source_subject=sha(SOURCE/('head_activation_witness.py' if a.mode=='witness' else 'evaluate_direct_target_student.py'))
    subjects=dict(dataset=[final['training_manifest']['sha256']],
        fit=[final['fit_report']['sha256'],root['sha256'],owner['sha256']],
        export=[final['head']['sha256'],final['checkpoint']['sha256'],root['sha256']],source=[source_subject])
    reviews={name:role(config['reviews'][name],digests) for name,digests in subjects.items()}
    for entry in [root,owner,*reviews.values()]:add(pins,entry['path'],entry['sha256'])
    for path in (Path(__file__),BASE/'prepare_bound_launcher.py',BASE/'diagnostic_verdict.py',
                 BASE/'test_launch_helpers.py',BASE/'launch_helper_tests.xml',BASE/'source_stub_tests.xml',
                 BASE/'runtime_inventory.json',BASE/'source_derivation.json',BASE/'source_preparation.json',a.reviews):add(pins,path)
    b=dict(**final,reviews=reviews,root_audit=root,owner_completion=owner,producer_exit=producer,
        onnx_dependencies=inventory['onnx_dependencies'],training_dataset_sha256=[final['training_manifest']['sha256']],
        dataset_subject_semantics='completed full training manifest qualified by immutable final dataset review',
        recursive_training_hashes=False,runtime_inventory_sha256=sha(BASE/'runtime_inventory.json'),
        ordinary_final_step=55000,controller='direct_absolute_target_1000_applied_inverse_learned_prior',
        architecture=[1000,256,256,23],hidden_activation='ELU',head_output='normalized_target',
        span_contract='existing_float32_joint_span_promoted_float64',learned_BFM_calls=0,hardware_authorized=False,
        clock_foundation_connected=False,filters_enabled=False,compiled_preview_enabled=False,
        root_authorized_witness=True,root_authorized_evaluation=True,expected_head_calls=1,
        physics_authorized=a.mode=='evaluation',requested_main_controls=1569,conditional_hold_controls=250)
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
        write_new(frozen,dict(kind='one_final55000_canonical_direct_target_evaluation',source_sha256=current,input_sha256=pins,
            requested_main_controls=1569,conditional_hold_controls=250,learned_BFM_calls=0,
            recursive_training_hashes=False,hardware_authorized=False))
        add(pins,frozen)
    b['input_files']=[dict(path=p,sha256=d) for p,d in sorted(pins.items())]
    path=BASE/(a.mode+'_binding.json');write_new(path,b)
    require_model_ready(BASE,'witness') if a.mode=='witness' else require_ready(BASE)
    print(json.dumps(dict(binding_sha256=sha(path),pins=len(pins),recursive_training_hashes=False,model_calls=0,native_steps=0)))

if __name__=='__main__':main()
