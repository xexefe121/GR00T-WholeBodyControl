"""Synthetic metadata tests only; never launches the actual fit."""
import copy
import pytest
from verify_completed import canonical_pins,validate_process_pin_report,validate_report,expected,sha

def report_fixture():
    counts=dict(training=expected(9000,44058000),diagnostics={name:expected(1437,367570) for name in
        ('initial_GPU32','final_GPU32','CPU64','GPU64','ORT64')},calibration_forward_calls=0,calibration_gradient_calls=0,
        BFM_calls=0,native_calls=0,manual_export_trace_calls=0)
    return dict(completed=True,optimization_completed=True,final_export_diagnostics_completed=True,numerical_gate_passed=True,
        export_parity_passed=True,all_frozen_inputs_unchanged=True,context_and_normalization_reused=True,
        condition='causal',ordinary_start_step=68000,ordinary_final_step=71000,additional_updates=3000,
        optimizer_start_step=3000,optimizer_step=6000,fresh_optimizer=False,features=1323,context_features=323,
        head_output='normalized_target',fixed_full_state_coefficient=1.8188207859141674,execution_dtype='float64',
        public_input_dtype='float32',public_output_dtype='float32',parity_tolerance_rad=1e-5,
        checkpoint_selection=False,native_steps=0,BFM_calls=0,training_first_layer_execution='split_contiguous_1000_plus_323',
        export_first_layer_execution='monolithic_float64_1323',counts=counts,max_preclip_error_rad=1e-8,
        initial_restoration={'passed':True})

def test_owner_accepts_complete_fixed_warm_counts():assert validate_report(report_fixture())

@pytest.mark.parametrize('key,value',[('fresh_optimizer',True),('optimizer_step',3000),('ordinary_final_step',68000),('max_preclip_error_rad',2e-5)])
def test_owner_rejects_wrong_warm_or_gate_metadata(key,value):
    value_report=report_fixture();value_report[key]=value
    with pytest.raises(ValueError):validate_report(value_report)

def test_owner_retains_truthful_failed_counter_prefix():
    value=report_fixture();value['completed']=False;value['counts']['training']=expected(3,14686)
    assert not validate_report(value)
    value['counts']['training']['calls_verified']=4
    with pytest.raises(ValueError):validate_report(value)

def test_pin_membership_rejects_omission_and_duplicate(tmp_path):
    a=tmp_path/'one';a.write_text('first');b=tmp_path/'two';b.write_text('second')
    pins={str(a):sha(a),str(b):sha(b)}
    record=dict(all_exact=True,count=2,files={p:dict(expected=h,actual=h,matched=True) for p,h in pins.items()})
    validate_process_pin_report(record,pins)
    corrupt=copy.deepcopy(record);del corrupt['files'][str(b)]
    with pytest.raises(ValueError):validate_process_pin_report(corrupt,pins)
    with pytest.raises(ValueError):canonical_pins({str(a):sha(a),str(a).upper():sha(a)})

def test_pin_record_does_not_accept_changed_current_file(tmp_path):
    a=tmp_path/'one';a.write_text('first');digest=sha(a)
    record=dict(all_exact=True,count=1,files={str(a):dict(expected=digest,actual=digest,matched=True)})
    a.write_text('changed')
    with pytest.raises(ValueError):validate_process_pin_report(record,{str(a):digest})

