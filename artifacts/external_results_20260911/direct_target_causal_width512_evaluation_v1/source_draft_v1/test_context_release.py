"""Reject incomplete or mismatched endpoint evidence before any model is created."""
import copy
import pytest
from context_release import condition_report, counter, COEFFICIENT, GROUP_WEIGHTS, WEIGHT_RULE, SOURCE_CHECKPOINT


def report(condition):
    return dict(completed=True,optimization_completed=True,final_export_diagnostics_completed=True,
        numerical_gate_passed=True,export_parity_passed=True,condition=condition,ordinary_final_step=81000,
        additional_updates=10000,optimizer_step=16000,fresh_optimizer=False,features=1323,context_features=323,
        head_output='normalized_target',fixed_full_state_coefficient=COEFFICIENT,execution_dtype='float64',
        ordinary_start_step=71000,optimizer_start_step=6000,context_and_normalization_reused=True,
        architecture=[1323,512,512,23],hidden_width=512,expansion_seed=20260912,
        training_first_layer_execution='split_old256_new256_original1000_plus323',export_first_layer_execution='monolithic_float64_1323',
        response_group_weights=list(GROUP_WEIGHTS),response_weight_rule=WEIGHT_RULE,source_checkpoint_sha256=SOURCE_CHECKPOINT,
        public_input_dtype='float32',public_output_dtype='float32',parity_tolerance_rad=1e-5,
        all_frozen_inputs_unchanged=True,checkpoint_selection=False,native_steps=0,BFM_calls=0,
        max_preclip_error_rad=0.,counts=dict(training=counter(30000,146860000),
            diagnostics={k:counter(1437,367570) for k in ('initial_GPU32','final_GPU32','CPU64','GPU64','ORT64')},
            calibration_forward_calls=0,calibration_gradient_calls=0,BFM_calls=0,native_calls=0,manual_export_trace_calls=0))


@pytest.mark.parametrize('condition',['causal'])
def test_fixed_complete_condition_schema(condition):
    condition_report(report(condition),condition)


@pytest.mark.parametrize('field,value',[
    ('condition','blinded'),('ordinary_final_step',71000),('features',1000),
    ('architecture',[1323,256,256,23]),('hidden_width',256),('expansion_seed',0),('additional_updates',3000),
    ('training_first_layer_execution','monolithic'),('export_first_layer_execution','float32'),
    ('ordinary_start_step',65000),('optimizer_start_step',0),('optimizer_step',3000),('fresh_optimizer',True),
    ('context_and_normalization_reused',False),('response_group_weights',[1.]*6),
    ('response_weight_rule','arbitrary'),('source_checkpoint_sha256','0'*64),
    ('completed',False),('max_preclip_error_rad',1.000001e-5),('max_preclip_error_rad',float('nan')),
    ('checkpoint_selection',True),('fixed_full_state_coefficient',10000.)])
def test_wrong_endpoint_or_failed_numerics_rejected(field,value):
    value_report=report('causal');value_report[field]=value
    with pytest.raises(AssertionError):condition_report(value_report,'causal')


@pytest.mark.parametrize('kind',['unreturned_call','extra_training_row','missing_diagnostic','extra_calibration'])
def test_incomplete_or_extra_work_cannot_pass(kind):
    value=report('causal')
    if kind=='unreturned_call':value['counts']['training']['calls_returned']-=1
    if kind=='extra_training_row':value['counts']['training']['rows_attempted']+=1
    if kind=='missing_diagnostic':del value['counts']['diagnostics']['GPU64']
    if kind=='extra_calibration':value['counts']['calibration_gradient_calls']=1
    with pytest.raises(AssertionError):condition_report(value,'causal')
