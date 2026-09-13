"""Freeze actual reviewed direct-target exports only; no inference or physics."""
import argparse
import json
from pathlib import Path
import sys

BASE=Path(__file__).resolve().parent
SOURCE=BASE/'source_draft_v1'
sys.path.insert(0,str(SOURCE))
from evaluation_gate import read,sha,field,has_hash,bound_file,require_model_ready,require_ready,CENTERS_SHA,LABEL_SHA

FIT=BASE.parent/'direct_target_student_v1'
ORIGINAL=BASE.parent/'fast_controller_phase_fit_v1'
DEPS=Path('E:/codex_sonic_runtime/bfm_seed_20260910/onnx_deps')
DATA_ROLES=('centers','velocity_features','velocity_target','physical_manifest','pico','walk002')

def item(path):
    path=Path(path).resolve()
    assert path.is_absolute() and path.is_file(),str(path)
    return dict(path=path.as_posix(),sha256=sha(path))

def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def add(pins,path,digest=None):
    entry=item(path)
    if digest is not None:assert entry['sha256']==digest,str(path)
    old=pins.get(entry['path'])
    assert old is None or old==entry['sha256'],str(path)
    pins[entry['path']]=entry['sha256']
    return entry

def role(spec,subjects):
    entry=item(spec['path']);entry['pass_field']=spec['pass_field']
    review=read(bound_file(entry))
    assert field(review,entry['pass_field']) is True
    assert subjects and all(has_hash(review,digest) for digest in subjects)
    return entry

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['witness','evaluation'],required=True)
    p.add_argument('--reviews',type=Path,required=True);a=p.parse_args()
    config=read(a.reviews)
    assert config['root_selected'] is True
    assert config['selected_witness_calls']==1 and config['selected_main_controls']==1569 and config['conditional_hold_controls']==250
    binding_path=BASE/(a.mode+'_binding.json');assert not binding_path.exists()
    absent=('head_witness',) if a.mode=='witness' else ('nominal','post_lifecycle_hold_5s','pilot_outcome.json')
    for name in absent:assert not (BASE/name).exists(),name
    request=read(FIT/'training_request.json');training=read(FIT/'training_frozen_inputs.json')
    assert request['ordinary_final_step']==request['updates']==5000 and request['root_selected'] is True
    assert request['architecture']==[1000,256,256,23] and request['head_output']=='normalized_target'
    pins={}
    final={k:add(pins,v) for k,v in dict(head=FIT/'fit/student_head.onnx',checkpoint=FIT/'fit/student_head.pt',
        fit_report=FIT/'fit/report.json',training_manifest=FIT/'training_frozen_inputs.json',
        training_request=FIT/'training_request.json',centers=request['paths']['centers'],
        contract=request['paths']['contract'],query250_labels=BASE.parent/'bfm_entry250_labels_v1/labels/labels.npz').items()}
    fit=read(bound_file(final['fit_report']))
    assert fit['ordinary_final_step']==fit['additional_updates']==5000
    assert fit['features']==1000 and fit['head_output']=='normalized_target'
    for key in ('optimization_completed','final_export_diagnostics_completed','numerical_gate_passed','export_parity_passed'):assert fit[key] is True,key
    assert fit['checkpoint_sha256']==final['checkpoint']['sha256'] and fit['onnx_sha256']==final['head']['sha256']
    assert final['centers']['sha256']==CENTERS_SHA and final['query250_labels']['sha256']==LABEL_SHA
    data=[add(pins,request['paths'][k]) for k in DATA_ROLES]
    root=role(config['root_audit'],[final['fit_report']['sha256'],final['head']['sha256'],final['checkpoint']['sha256']])
    # The completed numerical reviewer must bind the root audit and actual export directly.
    subjects=dict(dataset=[e['sha256'] for e in data],fit=[final['fit_report']['sha256'],root['sha256']],
        export=[final['head']['sha256'],final['checkpoint']['sha256'],root['sha256']],
        source=[sha(SOURCE/('head_activation_witness.py' if a.mode=='witness' else 'evaluate_direct_target_student.py'))])
    reviews={k:role(config['reviews'][k],v) for k,v in subjects.items()}
    # Preserve the reviewed evaluator source tree and all original inputs without importing controllers.
    preparation=read(BASE/'source_preparation.json');assert preparation['passed'] is True
    actual={p.relative_to(SOURCE).as_posix():sha(p) for p in SOURCE.rglob('*.py')}
    assert actual==preparation['source_sha256']
    for path,digest in preparation['evidence_sha256'].items():add(pins,path,digest)
    for path in SOURCE.rglob('*.py'):add(pins,path)
    original=read(ORIGINAL/'frozen_inputs_v2.json')
    for path,digest in original['input_sha256'].items():add(pins,path,digest)
    for path,digest in training['input_sha256'].items():add(pins,path,digest)
    for name,digest in training['source_sha256'].items():add(pins,Path(training['source_directory'])/name,digest)
    for path in DEPS.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.py','.so'):add(pins,path)
    for entry in [root,*reviews.values()]:add(pins,entry['path'],entry['sha256'])
    for path in (Path(__file__),BASE/'prepare_bound_launcher.py',BASE/'diagnostic_verdict.py',
                 BASE/'test_launch_helpers.py',BASE/'launch_helper_tests_final_v2.xml',BASE/'source_preparation.json',a.reviews):add(pins,path)
    # The producer exit remains independent of the root saved-evidence audit.
    producer_exit=add(pins,FIT/'fit_process/exit.json');exit_record=read(producer_exit['path'])
    assert exit_record['raw_python_exit_code']==exit_record['exit_code']==0
    assert exit_record['exit_known'] is True and exit_record['error'] is None and exit_record['all_postrun_pins_exact'] is True
    for name in ('prerun_pins.json','postrun_pins.json','start.json','child.json'):
        add(pins,FIT/'fit_process'/name)
    for name in ('prerun_pins.json','postrun_pins.json'):
        record=read(FIT/'fit_process'/name);assert record['all_exact'] is True
        for path,entry in record['files'].items():
            assert entry['matched'] is True and entry['expected']==entry['actual']
            add(pins,path,entry['expected'])
    binding=dict(**final,reviews=reviews,root_audit=root,producer_exit=producer_exit,
        onnx_dependencies=DEPS.as_posix(),training_dataset_sha256=[e['sha256'] for e in data],
        ordinary_final_step=5000,controller='direct_absolute_target_1000_applied_inverse_learned_prior',
        architecture=[1000,256,256,23],hidden_activation='ELU',head_output='normalized_target',
        span_contract='existing_float32_joint_span_promoted_float64',learned_BFM_calls=0,
        hardware_authorized=False,clock_foundation_connected=False,filters_enabled=False,compiled_preview_enabled=False,
        root_authorized_witness=True,root_authorized_evaluation=True,expected_head_calls=1,
        physics_authorized=a.mode=='evaluation',requested_main_controls=1569,conditional_hold_controls=250)
    if a.mode=='evaluation':
        for name,path in dict(first_export_witness=BASE/'head_witness/witness.npz',
                              first_export_receipt=BASE/'head_witness/report.json').items():binding[name]=add(pins,path)
        for name in ('witness_binding.json','witness_process/exit.json','witness_process/raw_exit.json',
                     'witness_process/diagnostic_verdict.json','witness_process/postrun_hashes.json',
                     'witness_process/launch_receipt.json','witness_process/launch_clearance.json'):
            add(pins,BASE/name)
        witness_exit=read(BASE/'witness_process/exit.json')
        assert witness_exit['exit_code']==0 and witness_exit['error'] is None
        assert read(BASE/'witness_process/raw_exit.json')['raw_python_exit_code']==0
        assert read(BASE/'witness_process/diagnostic_verdict.json')['passed'] is True
        frozen=BASE/'frozen_inputs_v2.json'
        write_new(frozen,dict(kind='one_fresh_direct_target_canonical_evaluation',source_sha256=actual,input_sha256=pins,
            requested_main_controls=1569,conditional_hold_controls=250,learned_BFM_calls=0,
            runtime_math_unchanged_from_reviewed_direct_adapter=True,hardware_authorized=False))
        add(pins,frozen)
    binding['input_files']=[dict(path=p,sha256=d) for p,d in sorted(pins.items())]
    # A failed validation keeps the attempted immutable binding; no overwrite or automatic retry.
    write_new(binding_path,binding)
    require_model_ready(BASE,'witness') if a.mode=='witness' else require_ready(BASE)
    print(json.dumps(dict(mode=a.mode,binding_sha256=sha(binding_path),pins=len(pins),inference_calls=0,physics_steps=0)))

if __name__=='__main__':main()
