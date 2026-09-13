"""Tiny saved-JSON owner fixtures; zero task arrays, model or native calls."""
import json
from pathlib import Path
from unittest.mock import patch
import pytest
import prepare_owner_views as module
from context_release import SUBJECTS,counter,COEFFICIENT


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path


def make_pair(base):
    shared=base/'fit/shared';process=base/'fit_process';common={}
    for key in SUBJECTS:
        if key not in ('fit_report','checkpoint','head','export_manifest','normalization','shared_manifest','context_alignment','paired_report','blinded_fit_report','causal_fit_report','training_manifest','training_request'):
            common[key]=put(base/(key+'.json'),{})
    common['normalization']=put(shared/'normalization.npz',{})
    common['context_alignment']=put(shared/'context_alignment.json',{})
    common['shared_manifest']=put(shared/'output_manifest.json',{'files':{p.name:module.sha(p) for p in (common['normalization'],common['context_alignment'])}})
    common['training_request']=put(base/'training_request.json',{})
    source=put(base/'source/unchanged.py',{})
    required={p.resolve().as_posix():module.sha(p) for key,p in common.items() if key not in ('normalization','context_alignment','shared_manifest')}
    frozen={'input_sha256':required,'source_directory':str(source.parent),'source_sha256':{source.name:module.sha(source)}}
    common['training_manifest']=put(base/'training_frozen_inputs.json',frozen)
    direct={'training_request':module.sha(common['training_request']),'training_frozen_inputs':module.sha(common['training_manifest']),
        'shared_normalization':module.sha(common['normalization']),'shared_manifest':module.sha(common['shared_manifest']),
        'source_checkpoint':module.sha(common['source_checkpoint'])}
    conditions={}
    for condition in ('blinded','causal'):
        dest=base/'fit'/condition;values=dict(common)
        values['checkpoint']=put(dest/'student_head.pt',{'condition':condition})
        values['head']=put(dest/'student_head.onnx',{'condition':condition})
        values['export_manifest']=put(dest/'output_manifest.json',{})
        report=dict(completed=True,optimization_completed=True,final_export_diagnostics_completed=True,numerical_gate_passed=True,
            export_parity_passed=True,condition=condition,ordinary_final_step=68000,additional_updates=3000,optimizer_step=3000,
            fresh_optimizer=True,features=1323,context_features=323,head_output='normalized_target',fixed_full_state_coefficient=COEFFICIENT,
            execution_dtype='float64',public_input_dtype='float32',public_output_dtype='float32',parity_tolerance_rad=1e-5,
            all_frozen_inputs_unchanged=True,checkpoint_selection=False,native_steps=0,BFM_calls=0,max_preclip_error_rad=0.,
            counts={'training':counter(9000,44058000),'diagnostics':{k:counter(1437,367570) for k in ('initial_GPU32','final_GPU32','CPU64','GPU64','ORT64')},
                **{k:0 for k in ('calibration_forward_calls','calibration_gradient_calls','BFM_calls','native_calls','manual_export_trace_calls')}})
        for key,role in (('checkpoint_sha256','checkpoint'),('onnx_sha256','head'),('normalization_sha256','normalization'),
            ('training_request_sha256','training_request'),('frozen_receipt_sha256','training_manifest'),
            ('output_manifest_sha256','export_manifest'),('shared_manifest_sha256','shared_manifest')):report[key]=module.sha(values[role])
        values['fit_report']=put(dest/'report.json',report);conditions[condition]=values
        for role,alias in [('fit_report','report'),('checkpoint','checkpoint'),('head','head'),('export_manifest','output_manifest')]:direct[condition+'_'+alias]=module.sha(values[role])
    pair=put(base/'fit/paired_report.json',{'completed':True,'conditions':['blinded','causal'],
        'condition_report_sha256':{c:direct[c+'_report'] for c in conditions}});direct['paired_report']=module.sha(pair)
    for values in conditions.values():
        values['paired_report']=pair
        for c in conditions:values[c+'_fit_report']=conditions[c]['fit_report']
    launcher=put(base/'run.ps1',{});review=put(base/'review.json',{})
    clear=put(base/'training_clearance.json',dict(request_sha256=direct['training_request'],frozen_receipt_sha256=direct['training_frozen_inputs'],
        review_path=str(review),review_sha256=module.sha(review),launcher_path=str(launcher),launcher_sha256=module.sha(launcher)))
    put(process/'start.json',dict(wrapper_pid=1,request_sha256=direct['training_request'],frozen_receipt_sha256=direct['training_frozen_inputs'],clearance_sha256=module.sha(clear)))
    put(process/'child.json',dict(wrapper_pid=1,child_pid=2,captured_handle_nonzero=True))
    exit_path=put(process/'exit.json',dict(wrapper_pid=1,child_pid=2,exit_known=True,child_started=True,raw_python_exit_code=0,exit_code=0,error=None,clearance_sha256=module.sha(clear)))
    direct['process_exit']=module.sha(exit_path)
    allpins=dict(required);allpins[source.resolve().as_posix()]=module.sha(source)
    for name in ('prerun_pins.json','postrun_pins.json'):put(process/name,dict(all_exact=True,files={p:dict(expected=h,actual=h,matched=True) for p,h in allpins.items()}))
    owner=put(base/'owner.json',dict(owner_verification_passed=True,accounting_passed=True,paired_completion_passed=True,numerical_completion_passed=True,
        raw_exit_known=True,raw_python_exit_code=0,exit_code=0,all_postrun_pins_exact=True,processes_absent=True,expected_pids=[1,2],
        current_pin_count=len(allpins),condition_completion={'blinded':True,'causal':True},direct_subject_sha256=direct))
    return owner,conditions


def run(base,owner,conditions):
    with patch.object(module,'FIT',base),patch.object(module,'actual_paths',side_effect=lambda c:conditions[c]):return module.build_views(owner)


def test_both_views_alias_exact18_without_selecting_condition(tmp_path):
    owner,conditions=make_pair(tmp_path);before=owner.read_bytes();views=run(tmp_path,owner,conditions)
    assert owner.read_bytes()==before
    for c,v in views.items():
        assert set(v['direct_subject_sha256'])==set(SUBJECTS) and len(v['direct_subject_sha256'])==18
        assert v['direct_subject_sha256']['head']==module.sha(conditions[c]['head'])
        assert v['parent_owner_subject']['sha256']==module.sha(owner)
        assert v['controller_selected'] is False and v['witness_selected'] is False


@pytest.mark.parametrize('mutation',['incomplete','unknown_exit','pid','postpin','head','coefficient','context','report','clearance'])
def test_mismatched_parent_or_alias_cannot_emit_view(tmp_path,mutation):
    owner,conditions=make_pair(tmp_path)
    if mutation in ('incomplete','unknown_exit','pid'):
        r=module.read(owner)
        if mutation=='incomplete':r['paired_completion_passed']=False
        elif mutation=='unknown_exit':r['raw_exit_known']=False
        else:r['expected_pids']=[1,3]
        put(owner,r)
    elif mutation=='postpin':
        p=tmp_path/'fit_process/postrun_pins.json';r=module.read(p);next(iter(r['files'].values()))['matched']=False;put(p,r)
    elif mutation=='head':put(conditions['causal']['head'],{'changed':True})
    elif mutation=='coefficient':put(conditions['causal']['coefficient'],{'changed':True})
    elif mutation=='context':put(conditions['causal']['context_alignment'],{'changed':True})
    elif mutation=='report':put(conditions['causal']['fit_report'],{'changed':True})
    else:put(tmp_path/'training_clearance.json',{'changed':True})
    with pytest.raises((AssertionError,KeyError)):run(tmp_path,owner,conditions)
