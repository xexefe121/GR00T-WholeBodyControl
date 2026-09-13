"""Synthetic bookkeeping tests only: no task checkpoint/model/process calls."""
import copy
import math
from pathlib import Path
import pytest
from verify_completed import expected,validate_counter,validate_report,BACKENDS
def report():
    return dict(completed=True,optimization_completed=True,final_export_diagnostics_completed=True,numerical_gate_passed=True,
        export_parity_passed=True,all_frozen_inputs_unchanged=True,fresh_optimizer=True,condition='blinded',ordinary_final_step=68000,
        additional_updates=3000,optimizer_step=3000,features=1323,context_features=323,head_output='normalized_target',
        fixed_full_state_coefficient=1.8188207859141674,execution_dtype='float64',public_input_dtype='float32',public_output_dtype='float32',
        parity_tolerance_rad=1e-5,max_preclip_error_rad=1e-8,checkpoint_selection=False,native_steps=0,BFM_calls=0,
        counts=dict(training=expected(9000,44058000),diagnostics={b:expected(1437,367570) for b in BACKENDS},
            calibration_forward_calls=0,calibration_gradient_calls=0,BFM_calls=0,native_calls=0,manual_export_trace_calls=0))
def test_complete_report_and_condition_identity():
    assert validate_report(report(),'blinded')
    with pytest.raises(ValueError):validate_report(report(),'causal')
def test_interrupted_counter_is_prefix_not_complete():
    value=expected(2,3);value['calls_attempted']=3;value['rows_attempted']=5
    validate_counter(value,3,5,False)
    with pytest.raises(ValueError):validate_counter(value,3,5,True)
@pytest.mark.parametrize('field,value',[('calls_returned',4),('calls_attempted',4),('rows_verified',6),('rows_attempted',-1)])
def test_invalid_counter_fails(field,value):
    counts=expected(3,5);counts[field]=value
    with pytest.raises(ValueError):validate_counter(counts,3,5,False)
@pytest.mark.parametrize('value',[math.nan,math.inf,1.0001e-5,-1.])
def test_numerical_failure_not_success(value):
    record=report();record['max_preclip_error_rad']=value
    with pytest.raises(ValueError):validate_report(record,'blinded')
def test_no_new_calibration_permitted():
    record=report();record['counts']['calibration_gradient_calls']=1
    with pytest.raises(ValueError):validate_report(record,'blinded')
def test_failure_can_preserve_partial_accounting():
    record=report();record['completed']=False;record['counts']['training']=expected(3,14686)
    assert validate_report(record,'blinded') is False
def test_launcher_uses_pair_verdict_and_original_guards():
    text=(Path(__file__).parent/'run_fit_durable_v1.ps1').read_text()
    for value in ('CreateNew','FileShare]::Delete','$capturedHandle=$child.Handle',"'fit/paired_report.json'",'raw_python_exit_code=$rawExit',"'train_context_pair.py'",'88116000',"@('blinded','causal')"):
        assert value in text
    assert 'Get-FileHash' not in text
    assert 'if($null -eq $rawExit)' in text
    assert text.index('$capturedHandle=$child.Handle')<text.index('$child.WaitForExit()')
