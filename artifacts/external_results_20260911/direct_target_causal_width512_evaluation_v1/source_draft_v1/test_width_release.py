"""Width release metadata and early activation gates; no models or task arrays."""
import copy,json
import pytest
import evaluation_gate
from context_release import training_request_identity,COEFFICIENT

def request():
    return dict(kind='causal_width512_warm_continuation',root_selected=True,condition='causal',
        updates=10000,ordinary_start_step=71000,ordinary_final_step=81000,optimizer_start_step=6000,optimizer_final_step=16000,
        fresh_optimizer=False,coefficient=COEFFICIENT,
        learning_rate=dict(ramp_updates=250,ramp_inclusive=[1e-6,1e-5],cosine_updates=9750,cosine_inclusive=[1e-5,1e-6]),features=1323,
        architecture=[1323,512,512,23],old_hidden_width=256,new_hidden_width=512,expansion_seed=20260912,
        old_optimizer_blocks_exact=True,new_optimizer_moments_zero=True,shared_optimizer_step_for_new_entries=6000,
        first_layer_execution='split_old256_new256_original1000_plus323',export_first_layer_execution='monolithic_float64_1323',
        context_order='previous_action23_then_incoming_history300',coefficient_recalibration=False,
        context_and_normalization_reused=True,initial_parity_tolerance_rad=1e-5,parity_tolerance_rad=1e-5,
        initial_byte_gate_required=False,automatic_retry=False,no_checkpoint_selection=True)

def test_exact_width_request():training_request_identity(request())

@pytest.mark.parametrize('key,value',[
    ('kind','causal_response_balanced_continuation'),('root_selected',False),('updates',3000),
    ('ordinary_start_step',68000),('ordinary_final_step',71000),('optimizer_start_step',0),('optimizer_final_step',6000),
    ('fresh_optimizer',True),('coefficient',1.),('architecture',[1323,256,256,23]),('new_hidden_width',256),
    ('expansion_seed',0),('old_optimizer_blocks_exact',False),('new_optimizer_moments_zero',False),
    ('shared_optimizer_step_for_new_entries',0),('context_order','current_history'),('context_and_normalization_reused',False),
    ('parity_tolerance_rad',1e-4),('initial_parity_tolerance_rad',1e-4),('automatic_retry',True)])
def test_wrong_width_request_rejected(key,value):
    r=request();r[key]=value
    with pytest.raises(AssertionError):training_request_identity(r)

@pytest.mark.parametrize('key,value',[('ramp_updates',251),('cosine_updates',10000),('ramp_inclusive',[1e-5,1e-5]),('cosine_inclusive',[1e-5,1e-7])])
def test_rate_schedule_is_exact(key,value):
    r=request();r['learning_rate'][key]=value
    with pytest.raises(AssertionError):training_request_identity(r)

def binding():
    r=dict(root_authorized_witness=True,ordinary_final_step=81000,
        controller='direct_absolute_target_1323_causal_width512',architecture=[1323,512,512,23],context_condition='causal',
        hidden_activation='ELU',head_output='normalized_target',span_contract='existing_float32_joint_span_promoted_float64',
        learned_BFM_calls=0,hardware_authorized=False,clock_foundation_connected=False,filters_enabled=False,
        compiled_preview_enabled=False,expected_head_calls=1,physics_authorized=False)
    r['head']={'sentinel':'no actual file or model'}
    return r

class BeforeModel(Exception):pass

def test_exact_width_activation_reaches_only_immutable_file_gate(tmp_path,monkeypatch):
    (tmp_path/'witness_binding.json').write_text(json.dumps(binding()))
    observed=[]
    def boundary(entry):observed.append(entry);raise BeforeModel()
    monkeypatch.setattr(evaluation_gate,'bound_file',boundary)
    with pytest.raises(BeforeModel):evaluation_gate.require_model_ready(tmp_path,'witness')
    assert observed==[binding()['head']]

@pytest.mark.parametrize('key,value',[('architecture',[1323,256,256,23]),('ordinary_final_step',71000),
    ('controller','direct_absolute_target_1323_causal_response_balanced'),('context_condition','blinded'),
    ('physics_authorized',True),('expected_head_calls',2)])
def test_wrong_width_activation_rejected_before_files(tmp_path,monkeypatch,key,value):
    r=binding();r[key]=value;(tmp_path/'witness_binding.json').write_text(json.dumps(r))
    def forbidden(entry):pytest.fail('Activation mismatch reached file gate')
    monkeypatch.setattr(evaluation_gate,'bound_file',forbidden)
    with pytest.raises(AssertionError):evaluation_gate.require_model_ready(tmp_path,'witness')
