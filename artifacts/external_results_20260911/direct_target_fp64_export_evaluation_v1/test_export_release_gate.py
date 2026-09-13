"""Tiny synthetic metadata only; zero checkpoint, ORT, native or optimizer calls."""
import copy,hashlib
from pathlib import Path
import pytest
from export_release_gate import validate_release

def fixture():
    digest=lambda name:hashlib.sha256(str(name).encode()).hexdigest()
    names=('head','checkpoint','fit_report','source_head','normalization','training_manifest','training_request','export_report','export_request','export_manifest')
    b={name:dict(path=name,sha256=digest(name)) for name in names};paths={name:name for name in names}
    b.update(release_kind='same55000_checkpoint_fp64_export',export_input_dtype='float32',export_output_dtype='float32',export_internal_dtype='float64')
    fit=dict(ordinary_final_step=55000,additional_updates=50000,optimization_completed=True,
        final_export_diagnostics_completed=True,completed=False,numerical_gate_passed=False,export_parity_passed=False,
        checkpoint_sha256=digest('checkpoint'),onnx_sha256=digest('source_head'),features=1000,head_output='normalized_target')
    ex=dict(ordinary_final_step=55000,optimizer_updates=0,features=1000,head_output='normalized_target',
        completed=True,numerical_gate_passed=True,export_parity_passed=True,checkpoint_sha256=digest('checkpoint'),
        source_fit_report_sha256=digest('fit_report'),source_onnx_sha256=digest('source_head'),normalization_sha256=digest('normalization'),
        onnx_sha256=digest('head'),parity_tolerance_rad=1e-5,max_preclip_error_rad=1e-6,
        public_input_dtype='float32',public_output_dtype='float32',execution_dtype='float64',
        ELU_implementation='Where(x>0,x,Exp(Min(x,0))-1)',optimization_completed_source=True,
        BFM_calls=0,native_steps=0,checkpoint_selection=False,all_frozen_inputs_unchanged=True)
    all_subjects={k:digest(k) for k in names}
    b['reviews']={}
    records=dict(fit_report=fit,export_report=ex)
    for role in ('root_training_audit','export_owner_completion','dataset','training','export','source'):
        entry=dict(path=role,sha256=digest(role),pass_field='passed')
        if role in ('root_training_audit','export_owner_completion'):b[role]=entry
        else:b['reviews'][role]=entry
        records[role]=dict(passed=True,direct_subject_sha256=dict(all_subjects,
            root_training_audit=digest('root_training_audit'),export_owner_completion=digest('export_owner_completion')))
    records['export_owner_completion'].update(raw_exit_known=True,raw_python_exit_code=0,exit_code=0,all_postrun_pins_exact=True,processes_absent=True)
    return b,paths,records,digest

def run(f):
    b,paths,records,digest=f
    return validate_release(b,paths,Path('source'),'witness',read=lambda path:records[str(path)],sha=digest,
        bound_file=lambda entry:entry['path'],field=lambda r,key:r[key],has_hash=lambda r,d:True)

def test_original_failure_preserved_by_new_release():
    f=fixture();before=copy.deepcopy(f[2]['fit_report']);assert run(f)==before
    assert f[2]['fit_report']==before and not before['completed']

@pytest.mark.parametrize('key',['completed','numerical_gate_passed','export_parity_passed'])
def test_original_failure_flags_cannot_be_rewritten(key):
    f=fixture();f[2]['fit_report'][key]=True
    with pytest.raises(AssertionError):run(f)

@pytest.mark.parametrize('key',['completed','numerical_gate_passed','export_parity_passed','all_frozen_inputs_unchanged'])
def test_new_export_must_pass_every_gate(key):
    f=fixture();f[2]['export_report'][key]=False
    with pytest.raises(AssertionError):run(f)

@pytest.mark.parametrize('key',['checkpoint_sha256','source_fit_report_sha256','source_onnx_sha256','normalization_sha256','onnx_sha256'])
def test_exact_source_and_export_subjects_required(key):
    f=fixture();f[2]['export_report'][key]='a'*64
    with pytest.raises(AssertionError):run(f)

def test_direct_training_subject_cannot_hide_elsewhere():
    f=fixture();receipt=f[2]['root_training_audit'];receipt['other_metadata']=receipt['direct_subject_sha256'].pop('checkpoint')
    with pytest.raises(AssertionError):run(f)

def test_positive_training_evidence_required():
    f=fixture();f[2]['root_training_audit']['passed']=False
    with pytest.raises(AssertionError):run(f)

def test_failed_original_head_cannot_replace_new_head():
    f=fixture();f[0]['head']=dict(f[0]['source_head'])
    with pytest.raises(AssertionError):run(f)

@pytest.mark.parametrize('value',[1.00000000001e-5,float('nan'),float('inf'),-1e-6])
def test_original_parity_threshold_unweakened(value):
    f=fixture();f[2]['export_report']['max_preclip_error_rad']=value
    with pytest.raises(AssertionError):run(f)

def test_exact_parity_boundary_allowed():
    f=fixture();f[2]['export_report']['max_preclip_error_rad']=1e-5;run(f)

@pytest.mark.parametrize('key',['raw_exit_known','all_postrun_pins_exact','processes_absent'])
def test_export_owner_must_finish_and_stop(key):
    f=fixture();f[2]['export_owner_completion'][key]=False
    with pytest.raises(AssertionError):run(f)

@pytest.mark.parametrize('key',['optimizer_updates','BFM_calls','native_steps'])
def test_export_has_no_extra_optimizer_or_physics(key):
    f=fixture();f[2]['export_report'][key]=1
    with pytest.raises(AssertionError):run(f)
