"""Width release metadata and early activation gates; no models or task arrays."""
import copy,json
import pytest
import evaluation_gate
from context_release import training_request_identity,COEFFICIENT


def binding():
    r=dict(root_authorized_witness=True,ordinary_final_step=91000,
        controller='direct_absolute_target_1323_causal_width512_recovery',architecture=[1323,512,512,23],context_condition='causal',
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
