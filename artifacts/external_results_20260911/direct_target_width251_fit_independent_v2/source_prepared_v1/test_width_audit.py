"""Synthetic saved-state/AST checks only: no model, optimizer or corpus loads."""
import ast,copy,math
from pathlib import Path
import numpy as np
import pytest
import torch
from audit_width_math import NAMES,OLD,NEW,GROUP,SEED,rate,source_errors,reconstruct_expansion,width_initial_errors,restoration_fields
from audit_restoration import differences

def source_fixture():
    actor={n:torch.full(s,(i+1)/100,dtype=torch.float32) for i,(n,s) in enumerate(zip(NAMES,OLD))}
    actor['0.weight'][0,0]=-0.0
    return dict(kind='direct_absolute_native23_target',condition='causal',ordinary_final_step=71000,
        optimizer_step=6000,context_blinded=False,full_state_coefficient=1.8188207859141674,
        context_order='previous_action23_then_incoming_history300',actor_state=actor,
        feature_mean=torch.zeros(1323),feature_std=torch.ones(1323),rng={'synthetic':torch.tensor([7,4],dtype=torch.uint8)},
        optimizer_state=dict(param_groups=[copy.deepcopy(GROUP)],state={i:dict(step=torch.tensor(6000.),
            exp_avg=torch.full(s,(i+1)/1000),exp_avg_sq=torch.full(s,(i+1)/10000)) for i,s in enumerate(OLD)}))

def independent_fixture(source):
    # A separate direct block-placement fixture, not the auditor constructor.
    blocks=[np.s_[:256,:],np.s_[:256],np.s_[:256,:256],np.s_[:256],np.s_[:,:256],np.s_[:]]
    actor={n:torch.zeros(s) for n,s in zip(NAMES,NEW)}
    for n,b in zip(NAMES,blocks):actor[n][b]=source['actor_state'][n]
    g=torch.Generator().manual_seed(20260912)
    for n,fan in [('0.weight',1323),('0.bias',1323),('2.weight',512),('2.bias',512)]:
        actor[n][256:].uniform_(-1/math.sqrt(fan),1/math.sqrt(fan),generator=g)
    opt=copy.deepcopy(source['optimizer_state'])
    for i,(s,b) in enumerate(zip(NEW,blocks)):
        for k in ('exp_avg','exp_avg_sq'):
            opt['state'][i][k]=torch.zeros(s);opt['state'][i][k][b]=source['optimizer_state']['state'][i][k]
    return dict(actor_state=actor,optimizer_state=opt,expansion_seed=20260912,expansion_generator_state=g.get_state(),
        feature_mean=source['feature_mean'].clone(),feature_std=source['feature_std'].clone(),
        rng_after_restoration=copy.deepcopy(source['rng']),ordinary_start_step=71000,optimizer_start_step=6000,hidden_width=512,condition='causal')

def test_expansion_saved_bytes_and_source_global_rng_unchanged():
    source=source_fixture();prior=copy.deepcopy(source);global_before=torch.get_rng_state().clone()
    actual=independent_fixture(source)
    assert width_initial_errors(actual,source)==[]
    assert differences(source,prior)==[] and differences(torch.get_rng_state(),global_before)==[]
    assert torch.signbit(actual['actor_state']['0.weight'][0,0])
    assert all(float(s['step'])==6000 for s in actual['optimizer_state']['state'].values())

@pytest.mark.parametrize('damage',['old_context','new_outgoing','random_incoming','random_bias','old_moment','new_moment','shared_step','rng','local_rng','seed','normalization','group'])
def test_saved_expansion_corruptions_rejected(damage):
    source=source_fixture();initial=independent_fixture(source)
    if damage=='old_context':initial['actor_state']['0.weight'][0,1200]=0
    elif damage=='new_outgoing':initial['actor_state']['2.weight'][0,256]=.01
    elif damage=='random_incoming':initial['actor_state']['2.weight'][256,0]=0
    elif damage=='random_bias':initial['actor_state']['0.bias'][256]=0
    elif damage=='old_moment':initial['optimizer_state']['state'][4]['exp_avg'][0,0]=0
    elif damage=='new_moment':initial['optimizer_state']['state'][2]['exp_avg_sq'][256,0]=.01
    elif damage=='shared_step':initial['optimizer_state']['state'][0]['step'].fill_(0)
    elif damage=='rng':initial['rng_after_restoration']['synthetic'][0]=0
    elif damage=='local_rng':initial['expansion_generator_state'][0]^=1
    elif damage=='seed':initial['expansion_seed']+=1
    elif damage=='normalization':initial['feature_std'][1000]=2
    elif damage=='group':initial['optimizer_state']['param_groups'][0]['lr']=1e-5
    assert width_initial_errors(initial,source)

@pytest.mark.parametrize('damage',['step','variance','dtype','order','group','blinded'])
def test_source_checkpoint_contract_corruption(damage):
    source=source_fixture()
    if damage=='step':source['optimizer_state']['state'][2]['step'].fill_(6000.5)
    elif damage=='variance':source['optimizer_state']['state'][1]['exp_avg_sq'][0]=-1
    elif damage=='dtype':source['actor_state']['0.weight']=source['actor_state']['0.weight'].double()
    elif damage=='order':source['actor_state']=dict(reversed(list(source['actor_state'].items())))
    elif damage=='group':source['optimizer_state']['param_groups'][0]['fused']=True
    elif damage=='blinded':source['context_blinded']=True
    assert source_errors(source)

def test_full_ramp_cosine_boundary_and_index_contract():
    values=np.array([rate(i) for i in range(10000)])
    assert values[0]==values[-1]==1e-6 and values[249]==values[250]==1e-5
    assert np.all(np.diff(values[:250])>0) and np.all(np.diff(values[250:])<0)
    for bad in (-1,10000,True,1.0):
        with pytest.raises(ValueError):rate(bad)
    assert restoration_fields()['shared_optimizer_step_for_new_entries']==6000

def main_tree():return ast.parse(Path(__file__).with_name('audit_saved_warm.py').read_text())

def test_only_initial_tolerance_gate_and_correct_final_step_width_counts():
    tree=main_tree();text=ast.unparse(tree)
    assert 'nominal_f32_mean' not in text
    assert "close('nominal_cells', progress[:, 0], cell_values['nominal'].mean(axis=1), nominal=True)" in text
    assert 'rtol=3e-07 if nominal else 5e-12' in text
    assert "check(condition + '_initial_drift_gate', maximum <= 1e-05)" in text
    assert "tree('retained_expansion_generator', checkpoint['expansion_generator_state'], init['expansion_generator_state'])" in text
    assert 'counts(30000, 146860000)' in text and 'schedule(10000)' in text
    assert 'float(state[\'step\']) == 16000' in text and 'range(10000)' in text
    assert "read(shared / 'schedule_lineage.json')" in text

def test_no_model_gradient_optimizer_or_native_calls_in_new_helper():
    tree=ast.parse(Path(__file__).with_name('audit_width_math.py').read_text())
    for n in ast.walk(tree):
        if isinstance(n,(ast.Import,ast.ImportFrom)):
            value=ast.unparse(n)
            assert not any(v in value for v in ('width512','torch.nn','torch.optim','onnxruntime','mujoco'))
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):assert n.func.attr not in ('forward','backward','step','load','load_state_dict','InferenceSession')

def test_width512_graph_rejects_old256_weights():
    from test_prior_math import graph_fixture
    from audit_graph import audit_graph
    g,c,n=graph_fixture()
    c['actor_state']['0.weight']=torch.zeros((256,1323))
    with pytest.raises(AssertionError):audit_graph(g,c,n)
