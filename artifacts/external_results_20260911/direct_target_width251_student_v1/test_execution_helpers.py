"""Pure synthetic JSON/process ledger regressions; no model or array imports."""
import copy
from pathlib import Path
import pytest
import verify_completed as v
from execution_common import expected

def request():
    return dict(updates=10000,ordinary_start_step=81000,ordinary_final_step=91000,optimizer_start_step=16000,optimizer_final_step=26000,
        condition='causal',automatic_retry=False,recovery_coefficient=0.2,budgets=dict(training_forward_calls=40000,training_forward_rows=157040000),
        runtime={'python_path':'E:/runtime/python.exe'},subjects={'checkpoint':{'path':'E:/fit/source.pt','sha256':'a'*64},'collection_report':{'path':'E:/collection/report.json','sha256':'b'*64}})
def report():
    r=request()
    value=dict(completed=True,optimization_completed=True,final_export_diagnostics_completed=True,numerical_gate_passed=True,export_parity_passed=True,
        all_frozen_inputs_unchanged=True,context_and_normalization_reused=True,condition='causal',ordinary_start_step=81000,ordinary_final_step=91000,additional_updates=10000,
        optimizer_start_step=16000,optimizer_step=26000,fresh_optimizer=False,features=1323,context_features=323,head_output='normalized_target',fixed_full_state_coefficient=1.8188207859141674,
        execution_dtype='float64',public_input_dtype='float32',public_output_dtype='float32',parity_tolerance_rad=1e-5,checkpoint_selection=False,native_steps=0,BFM_calls=0,
        training_first_layer_execution='split_old256_new256_original1000_plus323',hidden_width=512,architecture=[1323,512,512,23],export_first_layer_execution='monolithic_float64_1323',
        expansion_performed=False,recovery_rows=1018,recovery_phase_counts=[99,819,100],recovery_coefficient=0.2,source_checkpoint_sha256='a'*64,recovery_collection=r['subjects']['collection_report'],
        max_preclip_error_rad=1e-8,initial_restoration={'passed':True},direct_subject_sha256={k:x['sha256'] for k,x in r['subjects'].items()})
    value['counts']=dict(training=expected(40000,157040000),diagnostics={b:expected(1441,368588) for b in v.BACKENDS},
        calibration_forward_calls=0,calibration_gradient_calls=0,BFM_calls=0,native_calls=0,manual_export_trace_calls=0)
    return value
def test_actual_new_report_contract():assert v.validate_report(report(),request()) is True
@pytest.mark.parametrize('key,bad',[('ordinary_final_step',81000),('optimizer_step',16000),('expansion_performed',True),('recovery_rows',1017),('recovery_coefficient',0.21),('hidden_width',256),('parity_tolerance_rad',1.1e-5),('max_preclip_error_rad',float('nan')),('recovery_phase_counts',[100,818,100]),('fresh_optimizer',True),('features',1000),('native_steps',False)])
def test_contract_corruption(key,bad):
    value=report();value[key]=bad
    with pytest.raises(ValueError):v.validate_report(value,request())
@pytest.mark.parametrize('backend',v.BACKENDS)
def test_fourth_corpus_required(backend):
    value=report();value['counts']['diagnostics'][backend]=expected(1437,367570)
    with pytest.raises(ValueError):v.validate_report(value,request())
def test_three_forward_budget_rejected():
    value=report();value['counts']['training']=expected(30000,146860000)
    with pytest.raises(ValueError):v.validate_report(value,request())
def test_failed_prefix_accepts_attempt_without_return():
    value=report();value['completed']=False;value['counts']['training']=expected(0,0);value['counts']['training'].update(calls_attempted=1,rows_attempted=1018)
    assert v.validate_report(value,request()) is False
def test_unknown_setup_calls_rejected():
    with pytest.raises(ValueError):v.validate_report({'completed':False},request())
def test_explicit_zero_setup_failure():assert v.validate_report({'completed':False,'task_forward_calls':0,'optimizer_updates':0},request()) is False
def test_direct_subject_omission():
    value=report();value['direct_subject_sha256'].pop('collection_report')
    with pytest.raises(ValueError):v.validate_report(value,request())

def records(base):
    req=request();clear={'request_sha256':'c'*64,'frozen_receipt_sha256':'d'*64};launch='e'*64;clr='f'*64
    start=dict(wrapper_pid=123,request_sha256=clear['request_sha256'],frozen_receipt_sha256=clear['frozen_receipt_sha256'],clearance_sha256=clr,launch_receipt_sha256=launch,
        command=dict(python=req['runtime']['python_path'],arguments=['-u',str(base/'source_snapshot_v1/train_recovery.py')],working_directory=str(base/'source_snapshot_v1'),CUBLAS_WORKSPACE_CONFIG=':4096:8',threads=1))
    start.update({k:req[k] for k in ('ordinary_start_step','ordinary_final_step','optimizer_start_step','optimizer_final_step','condition','updates','budgets','automatic_retry')})
    child={'wrapper_pid':123,'child_pid':456,'captured_handle_nonzero':True}
    end=dict(wrapper_pid=123,child_pid=456,child_started=True,exit_known=True,raw_python_exit_code=0,exit_code=0,clearance_sha256=clr,launch_receipt_sha256=launch)
    dispatch=dict(wrapper_pid=123,captured_handle_nonzero=True,executable='C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe',clearance_sha256=clr,launch_receipt_sha256=launch,
        arguments=['-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',str(base/'run_fit_durable_v1.ps1'),'-ClearanceSha256',clr,'-LaunchReceiptSha256',launch])
    return start,child,end,dispatch,req,clear,launch,clr,base
def test_process_chain(tmp_path):assert v.validate_process(*records(tmp_path))==[123,456]
@pytest.mark.parametrize('index,key,bad',[(0,'wrapper_pid',124),(1,'wrapper_pid',124),(1,'captured_handle_nonzero',False),(2,'exit_known',False),(2,'raw_python_exit_code',None),(2,'raw_python_exit_code',True),(2,'child_started',False),(3,'wrapper_pid',124),(3,'launch_receipt_sha256','0'*64),(3,'executable','C:/Program Files/PowerShell/7/pwsh.exe')])
def test_process_contradictions(tmp_path,index,key,bad):
    args=list(records(tmp_path));args[index][key]=bad
    with pytest.raises(ValueError):v.validate_process(*args)
def test_launch_argument_missing(tmp_path):
    args=list(records(tmp_path));args[3]['arguments']=args[3]['arguments'][:-2]
    with pytest.raises(ValueError):v.validate_process(*args)
def test_static_pin_set(tmp_path):
    receipt={'input_sha256':{str(tmp_path/'input'):'a'*64},'source_directory':str(tmp_path/'source'),'source_sha256':{'driver.py':'b'*64}}
    clear={'request_sha256':'c'*64,'frozen_receipt_sha256':'d'*64}
    pins=v.static_pins(tmp_path,receipt,clear);assert len(pins)==4
    assert pins[(tmp_path/'training_frozen_inputs.json').as_posix()]=='d'*64
def test_duplicate_physical_pin(tmp_path):
    with pytest.raises(ValueError):v.canonical_pins({str(tmp_path/'foo'):'a'*64,str(tmp_path/'FOO'):'a'*64})
