"""Saved-only condition aliases for a completed pair; selects no controller.

The original pair owner is immutable. These views recheck its completed output
subjects and process/pin receipts, without traversing the training corpus again.
"""
import argparse
from pathlib import Path
from freeze_final_package import FIT,BASE,SUBJECTS,actual_paths,item,read,sha,write_new
from context_release import condition_report


def exact_true(value,names):
    for name in names:
        assert value[name] is True,name


def pin_receipt(value):
    assert value['all_exact'] is True and value['files']
    result={}
    for path,entry in value['files'].items():
        assert entry['matched'] is True and entry['actual']==entry['expected'],path
        result[Path(path).resolve().as_posix()]=entry['expected']
    return result


def build_views(owner_path):
    parent=item(owner_path);owner=read(owner_path)
    exact_true(owner,('owner_verification_passed','accounting_passed','paired_completion_passed',
        'numerical_completion_passed','raw_exit_known','all_postrun_pins_exact','processes_absent'))
    assert type(owner['raw_python_exit_code']) is int and owner['raw_python_exit_code']==0
    assert type(owner['exit_code']) is int and owner['exit_code']==0
    assert owner['condition_completion']=={'blinded':True,'causal':True}
    direct=owner['direct_subject_sha256'];process=FIT/'fit_process'
    evidence={name:item(process/name) for name in ('start.json','child.json','exit.json','prerun_pins.json','postrun_pins.json')}
    start=read(process/'start.json');child=read(process/'child.json');exit_record=read(process/'exit.json')
    pids=owner['expected_pids']
    assert len(pids)==2 and all(type(x) is int and x>0 for x in pids)
    assert pids==[start['wrapper_pid'],child['child_pid']]==[exit_record['wrapper_pid'],exit_record['child_pid']]
    assert child['wrapper_pid']==pids[0] and child['captured_handle_nonzero'] is True
    exact_true(exit_record,('exit_known','child_started'))
    assert type(exit_record['raw_python_exit_code']) is int and exit_record['raw_python_exit_code']==0
    assert type(exit_record['exit_code']) is int and exit_record['exit_code']==0 and exit_record['error'] is None
    assert direct['process_exit']==evidence['exit.json']['sha256']
    pre=pin_receipt(read(process/'prerun_pins.json'));post=pin_receipt(read(process/'postrun_pins.json'))
    assert pre==post
    frozen=read(FIT/'training_frozen_inputs.json')
    required={Path(p).resolve().as_posix():v for p,v in frozen['input_sha256'].items()}
    required.update({(Path(frozen['source_directory'])/p).resolve().as_posix():v for p,v in frozen['source_sha256'].items()})
    assert all(pre.get(p)==v for p,v in required.items())
    assert len(required)==owner['current_pin_count']
    clear=item(FIT/'training_clearance.json');clear_value=read(clear['path'])
    assert start['clearance_sha256']==exit_record['clearance_sha256']==clear['sha256']
    for filename,key in (('training_request.json','request_sha256'),('training_frozen_inputs.json','frozen_receipt_sha256')):
        assert sha(FIT/filename)==start[key]==clear_value[key]
    assert sha(clear_value['review_path'])==clear_value['review_sha256']
    assert sha(clear_value['launcher_path'])==clear_value['launcher_sha256']
    evidence['training_clearance.json']=clear
    # Actual immutable outputs, not merely producer assertions, resolve each alias.
    views={}
    for condition in ('blinded','causal'):
        paths=actual_paths(condition);subjects={name:item(paths[name]) for name in SUBJECTS}
        aliases={'fit_report':condition+'_report','checkpoint':condition+'_checkpoint','head':condition+'_head',
            'normalization':'shared_normalization','training_manifest':'training_frozen_inputs',
            'training_request':'training_request','export_manifest':condition+'_output_manifest',
            'source_checkpoint':'source_checkpoint','paired_report':'paired_report','shared_manifest':'shared_manifest',
            'blinded_fit_report':'blinded_report','causal_fit_report':'causal_report'}
        for role,alias in aliases.items():assert subjects[role]['sha256']==direct[alias],role
        for name in ('coefficient','source_checkpoint','full_state_generation_request','full_state_generation_report','full_state_data_audit','full_state_data_owner'):
            assert required.get(subjects[name]['path'])==subjects[name]['sha256'],name
        shared=read(paths['shared_manifest'])['files']
        assert shared['context_alignment.json']==subjects['context_alignment']['sha256']
        assert shared['normalization.npz']==subjects['normalization']['sha256']
        report=read(paths['fit_report']);condition_report(report,condition)
        for key,role in (('checkpoint_sha256','checkpoint'),('onnx_sha256','head'),('normalization_sha256','normalization'),
            ('training_request_sha256','training_request'),('frozen_receipt_sha256','training_manifest'),
            ('output_manifest_sha256','export_manifest'),('shared_manifest_sha256','shared_manifest')):
            assert report[key]==subjects[role]['sha256'],key
        pair=read(paths['paired_report']);assert pair['completed'] is True and pair['conditions']==['blinded','causal']
        assert pair['condition_report_sha256']=={c:subjects[c+'_fit_report']['sha256'] for c in ('blinded','causal')}
        views[condition]=dict(owner_verification_passed=True,condition=condition,parent_owner_subject=parent,
            subjects=subjects,direct_subject_sha256={k:v['sha256'] for k,v in subjects.items()},
            raw_exit_known=True,raw_python_exit_code=0,exit_code=0,all_postrun_pins_exact=True,processes_absent=True,
            expected_pids=pids,process_absence_from_bound_parent_owner=True,process_evidence=evidence,
            pre_and_post_pin_records_exact=True,training_pin_count=len(required),
            original_pair_owner_preserved=True,controller_selected=False,witness_selected=False,
            task_model_calls=0,native_steps=0,optimizer_updates=0,
            limitations=['Saved accounting aliases only; process absence is the bound completed parent observation.',
                'Training corpus pins are checked against completed pre/post and parent evidence, not rehashed recursively.',
                'No condition, witness or controller is selected by either owner view.'])
    assert sha(owner_path)==parent['sha256']
    return views


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pair-owner',type=Path,required=True);args=parser.parse_args()
    destinations={c:BASE/('fit_owner_'+c+'_v1.json') for c in ('blinded','causal')}
    assert all(not p.exists() for p in destinations.values())
    views=build_views(args.pair_owner)
    for condition,path in destinations.items():write_new(path,views[condition])
    print({c:sha(p) for c,p in destinations.items()})


if __name__=='__main__':main()
