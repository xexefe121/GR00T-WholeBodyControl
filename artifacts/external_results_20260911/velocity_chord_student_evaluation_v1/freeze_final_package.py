"""Bind real reviewed final artifacts; never create placeholder launch bindings.

Run only when the four completed review records exist. The configuration contains
their existing absolute paths and exact boolean pass-field names. This utility
does no inference or physics and does not launch the generated scripts.
"""
import argparse
import json
from pathlib import Path
import sys

BASE=Path(__file__).parent;SOURCE=BASE/'source_snapshot_v1'
sys.path.insert(0,str(SOURCE))
from evaluation_gate import bound_file,field,has_hash,read,sha,require_ready
from head_activation_witness import CENTERS_SHA,LABEL_SHA,require_witness_ready

FIT=BASE.parent/'velocity_chord_student_v1'
ORIGINAL=BASE.parent/'fast_controller_phase_fit_v1'
DEPS=Path('E:/codex_sonic_runtime/bfm_seed_20260910/onnx_deps')


def item(path):
    path=Path(path).resolve();assert path.is_file(),str(path)
    return dict(path=path.as_posix(),sha256=sha(path))


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['witness','evaluation'],required=True)
    parser.add_argument('--reviews',type=Path,required=True)
    args=parser.parse_args();config=read(args.reviews)
    assert config['root_selected'] is True
    binding_path=BASE/(args.mode+'_binding.json');assert not binding_path.exists()
    for name in ('head_witness',) if args.mode=='witness' else ('nominal','post_lifecycle_hold_5s','pilot_outcome.json'):
        assert not (BASE/name).exists(),name
    final={name:item(path) for name,path in dict(head=FIT/'fit/student_head.onnx',checkpoint=FIT/'fit/student_head.pt',
        fit_report=FIT/'fit/report.json',training_manifest=FIT/'training_frozen_inputs.json',
        centers=FIT/'generation/centers.npz',query250_labels=BASE.parent/'bfm_entry250_labels_v1/labels/labels.npz').items()}
    fit=read(bound_file(final['fit_report']))
    assert fit['ordinary_final_step']==70000 and fit['additional_updates']==5000
    for name in ('optimization_completed','final_export_diagnostics_completed','numerical_gate_passed','export_parity_passed'):assert fit[name] is True,name
    assert fit['onnx_sha256']==final['head']['sha256'] and fit['checkpoint_sha256']==final['checkpoint']['sha256']
    assert final['centers']['sha256']==CENTERS_SHA and final['query250_labels']['sha256']==LABEL_SHA
    driver=SOURCE/('head_activation_witness.py' if args.mode=='witness' else 'evaluate_velocity_chord_student.py')
    subjects=dict(dataset=CENTERS_SHA,fit=final['fit_report']['sha256'],export=final['head']['sha256'],source=sha(driver))
    reviews={}
    for role,digest in subjects.items():
        spec=config['reviews'][role];entry=item(spec['path']);entry['pass_field']=spec['pass_field']
        record=read(bound_file(entry));assert field(record,entry['pass_field']) is True and has_hash(record,digest),role
        reviews[role]=entry
    originals=read(ORIGINAL/'frozen_inputs_v2.json')
    pins=dict(originals['input_sha256'])
    for name,digest in originals['source_sha256'].items():assert sha(SOURCE/name)==digest,name
    for path,digest in pins.items():assert sha(path)==digest,path
    training=read(FIT/'training_frozen_inputs.json')
    for path,digest in training['input_sha256'].items():assert sha(path)==digest; pins[path]=digest
    for name,digest in training['source_sha256'].items():
        path=Path(training['source_directory'])/name;assert sha(path)==digest;pins[path.as_posix()]=digest
    for path in SOURCE.rglob('*.py'):pins[path.as_posix()]=sha(path)
    for path in DEPS.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.py','.so'):pins[path.as_posix()]=sha(path)
    for entry in list(final.values())+list(reviews.values()):pins[entry['path']]=entry['sha256']
    for path in (Path(__file__),BASE/'prepare_bound_launcher.py',BASE/'original_sources.json',BASE/'derive_evaluator.py',
                 BASE/'evaluator_derivation.json',BASE/'test_evaluation_package.py',BASE/'focused_final.xml',args.reviews):
        pins[item(path)['path']]=sha(path)
    binding=dict(**final,reviews=reviews,input_files=[dict(path=path,sha256=digest) for path,digest in sorted(pins.items())],
        onnx_dependencies=DEPS.as_posix(),training_dataset_sha256=[CENTERS_SHA],
        ordinary_final_step=70000,hardware_authorized=False)
    if args.mode=='witness':
        binding.update(root_authorized_single_head_witness=True,expected_head_calls=1,
            BFM_inference_authorized=False,physics_authorized=False,fitting_authorized=False)
        write_new(binding_path,binding);require_witness_ready(BASE)
    else:
        witness=BASE/'head_witness/witness.npz';receipt=BASE/'head_witness/report.json'
        binding.update(first_export_witness=item(witness),first_export_receipt=item(receipt),
            root_authorized_canonical_evaluation=True,requested_main_controls=1569,conditional_hold_controls=250,
            controller='original_unfiltered_raw_combined_action',filters_enabled=False,compiled_preview_enabled=False)
        for path in (witness,receipt,BASE/'witness_binding.json'):
            entry=item(path);pins[entry['path']]=entry['sha256']
        frozen=BASE/'frozen_inputs_v2.json'
        write_new(frozen,dict(kind='one_final70000_canonical_unfiltered_evaluation',
            source_sha256={path.relative_to(SOURCE).as_posix():sha(path) for path in SOURCE.rglob('*.py')},input_sha256=pins,
            no_runtime_math_changes=True,raw_combined_action_history_preserved=True,hardware_authorized=False))
        pins[frozen.as_posix()]=sha(frozen)
        binding['input_files']=[dict(path=path,sha256=digest) for path,digest in sorted(pins.items())]
        write_new(binding_path,binding);require_ready(BASE)
    print(json.dumps(dict(mode=args.mode,binding_sha256=sha(binding_path),pins=len(binding['input_files']),
        preparation_only=True,inference_calls=0,physics_steps=0)))


if __name__=='__main__':main()
