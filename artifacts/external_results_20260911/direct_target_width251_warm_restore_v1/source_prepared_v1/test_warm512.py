"""CPU-only synthetic tensor states. No checkpoint, forward, or optimizer step."""
import ast
import copy
import random
from pathlib import Path
import numpy as np
import pytest
import torch
from width512 import WiderContextTarget,NAMES,NEW_SHAPES
from balance_contract import GROUP_WEIGHTS,WEIGHT_RULE
from restoration_support import exact_saved,restore_rng as restore_global_rng
import warm512_restore as warm

torch.set_num_threads(1)

def rng_state():
    n=np.random.get_state()
    return dict(torch_cpu=torch.get_rng_state().clone(),torch_cuda=[],numpy=dict(name=n[0],keys=n[1].tolist(),position=n[2],has_gauss=n[3],cached_gaussian=n[4]),python=random.getstate())

def model_optimizer():
    m=WiderContextTarget(np.zeros(1323,np.float32),np.ones(1323,np.float32))
    o=torch.optim.AdamW(m.actor.parameters(),lr=1e-6,weight_decay=1e-5,foreach=False,fused=False)
    return m,o

def saved_state():
    actor={name:torch.full(shape,(i+1)*.001,dtype=torch.float32) for i,(name,shape) in enumerate(zip(NAMES,NEW_SHAPES))}
    actor['0.weight'][0,0]=-0.
    _,o=model_optimizer();optimizer=o.state_dict()
    optimizer['state']={i:dict(step=torch.tensor(16000.,dtype=torch.float32),exp_avg=torch.full(shape,(i+1)*.002),
                                  exp_avg_sq=torch.full(shape,(i+1)*.003)) for i,shape in enumerate(NEW_SHAPES)}
    return dict(kind='direct_absolute_native23_target',condition='causal',ordinary_final_step=81000,optimizer_step=16000,
        fresh_optimizer=False,hidden_width=512,architecture=[1323,512,512,23],context_blinded=False,
        context_order='previous_action23_then_incoming_history300',full_state_coefficient=warm.COEFFICIENT,
        response_group_weights=GROUP_WEIGHTS,response_weight_rule=WEIGHT_RULE,
        request=dict(ordinary_final_step=81000,optimizer_final_step=16000,architecture=[1323,512,512,23],
                     first_layer_execution=warm.FORWARD,export_first_layer_execution='monolithic_float64_1323'),
        actor_state=actor,optimizer_state=optimizer,feature_mean=torch.zeros(1323),feature_std=torch.ones(1323),rng=rng_state())

@pytest.fixture(autouse=True)
def cpu_rng_only(monkeypatch):
    original=rng_state()
    monkeypatch.setattr(torch.cuda,'device_count',lambda:0)
    def restore_cuda(values):assert values==[]
    monkeypatch.setattr(torch.cuda,'set_rng_state_all',restore_cuda)
    # Any accidental forward or update during restoration is a test failure.
    monkeypatch.setattr(WiderContextTarget,'forward',lambda *a,**k:pytest.fail('Unexpected synthetic model forward'))
    monkeypatch.setattr(torch.optim.AdamW,'step',lambda *a,**k:pytest.fail('Unexpected optimizer update'))
    yield
    restore_global_rng(original)

def test_full_warm_restoration_preserves_trained_context_and_added_blocks():
    saved=saved_state();before=copy.deepcopy(saved);model,opt=model_optimizer();progress={}
    torch.manual_seed(987);np.random.seed(987);random.seed(987)
    result=warm.restore(model,opt,saved,rng_state,progress)
    assert result['restoration_completed'] and progress['restoration_verified']
    assert result['optimizer_start_step']==16000 and result['expansion_performed'] is False
    assert result['weights_or_moments_zeroed'] is False and result['learning_rate_override_performed'] is False
    exact_saved(model.actor.state_dict(),saved['actor_state'])
    exact_saved(opt.state_dict(),saved['optimizer_state'])
    exact_saved(rng_state(),saved['rng']);exact_saved(saved,before)
    assert torch.count_nonzero(model.actor[2].weight[:256,256:])==256*256
    assert torch.count_nonzero(model.actor[4].weight[:,256:])==23*256
    assert torch.count_nonzero(model.actor[0].weight[:,1000:])==512*323

def test_loaded_state_does_not_alias_source_on_later_tensor_mutation():
    saved=saved_state();before=copy.deepcopy(saved);m,o=model_optimizer();warm.restore(m,o,saved,rng_state,{})
    with torch.no_grad():m.actor[0].weight[0,1]+=1
    o.state[next(iter(m.actor.parameters()))]['exp_avg'][0,1]+=1
    o.state[next(iter(m.actor.parameters()))]['step']+=1
    exact_saved(saved,before)

@pytest.mark.parametrize('field,value',[
    ('ordinary_final_step',71000),('optimizer_step',6000),('architecture',[1323,256,256,23]),
    ('hidden_width',256),('context_blinded',True),('fresh_optimizer',True),
    ('full_state_coefficient',1.),('response_weight_rule','Emean/Eg'),('ordinary_final_step',True)])
def test_rejects_old_or_changed_source_metadata(field,value):
    saved=saved_state();saved[field]=value
    with pytest.raises(ValueError):warm.validate_source(saved)

@pytest.mark.parametrize('field,value',[
    ('first_layer_execution','monolithic_float32_1323'),('export_first_layer_execution','float32'),
    ('ordinary_final_step',71000),('optimizer_final_step',6000)])
def test_requires_original_forward_and_source_request(field,value):
    saved=saved_state();saved['request'][field]=value
    with pytest.raises(ValueError):warm.validate_source(saved)

@pytest.mark.parametrize('corruption',['actor_shape','actor_order','actor_dtype','nonfinite','norm','state_missing','negative_moment','step','group_lr','group_flag','ids'])
def test_source_tensor_optimizer_schema_failures(corruption):
    s=saved_state();state=s['optimizer_state']['state'][0];group=s['optimizer_state']['param_groups'][0]
    if corruption=='actor_shape':s['actor_state']['0.weight']=torch.zeros(256,1323)
    if corruption=='actor_order':s['actor_state']=dict(reversed(list(s['actor_state'].items())))
    if corruption=='actor_dtype':s['actor_state']['0.bias']=s['actor_state']['0.bias'].double()
    if corruption=='nonfinite':s['actor_state']['4.bias'][0]=float('nan')
    if corruption=='norm':s['feature_std'][0]=0
    if corruption=='state_missing':state.pop('exp_avg')
    if corruption=='negative_moment':state['exp_avg_sq'][0,0]=-1
    if corruption=='step':state['step']=torch.tensor(15999.)
    if corruption=='group_lr':group['lr']=1e-5
    if corruption=='group_flag':group['foreach']=True
    if corruption=='ids':group['params']=[1,0,2,3,4,5]
    with pytest.raises(ValueError):warm.validate_source(s)

@pytest.mark.parametrize('corruption',['cpu_encoding','numpy_keys','numpy_position','python_state'])
def test_source_rng_schema_failures(corruption):
    s=saved_state()
    if corruption=='cpu_encoding':s['rng']['torch_cpu']=torch.zeros(2,dtype=torch.uint8)
    if corruption=='numpy_keys':s['rng']['numpy']['keys'][0]=-1
    if corruption=='numpy_position':s['rng']['numpy']['position']=625
    if corruption=='python_state':s['rng']['python']=(0,(),None)
    with pytest.raises(ValueError):warm.validate_source(s)

def test_validation_leaves_all_global_rng_unchanged():
    s=saved_state();before=rng_state();warm.validate_source(s);exact_saved(rng_state(),before)

@pytest.mark.parametrize('corruption',['norm','parameter_order','elu','forward_hook','instance_forward','frozen_param'])
def test_actual_destination_contract_rejected_before_actor_load(corruption):
    s=saved_state();m,o=model_optimizer();p={}
    if corruption=='norm':m.feature_mean[0]=1
    if corruption=='parameter_order':o.param_groups[0]['params'].reverse()
    if corruption=='elu':m.actor[1]=torch.nn.ReLU()
    if corruption=='forward_hook':m.actor[0].register_forward_hook(lambda *a:None)
    if corruption=='instance_forward':m.actor[0].forward=lambda x:x
    if corruption=='frozen_param':m.actor[0].weight.requires_grad_(False)
    with pytest.raises(ValueError):warm.restore(m,o,s,rng_state,p)
    assert not p['actor_load_attempted'] and not p['restoration_verified']

def test_optimizer_load_failure_preserves_partial_stage_without_retry(monkeypatch):
    s=saved_state();before=copy.deepcopy(s);m,o=model_optimizer();p={};calls=[]
    original=o.load_state_dict
    def failed(value):
        calls.append(1);original(value);raise RuntimeError('injected after optimizer mutation')
    monkeypatch.setattr(o,'load_state_dict',failed)
    with pytest.raises(RuntimeError,match='injected'):warm.restore(m,o,s,rng_state,p)
    assert calls==[1] and p['actor_load_returned'] and p['optimizer_load_attempted']
    assert not p['optimizer_load_returned'] and not p['rng_restore_attempted'] and not p['restoration_verified']
    exact_saved(s,before);exact_saved(m.actor.state_dict(),s['actor_state'])

def test_rng_failure_preserves_returned_loads_and_no_false_completion(monkeypatch):
    s=saved_state();m,o=model_optimizer();p={}
    def failed(_):raise RuntimeError('injected RNG failure')
    monkeypatch.setattr(warm,'restore_rng',failed)
    with pytest.raises(RuntimeError,match='RNG'):warm.restore(m,o,s,rng_state,p)
    assert p['actor_load_returned'] and p['optimizer_load_returned'] and p['rng_restore_attempted']
    assert not p['rng_restore_returned'] and not p['restoration_verified']

def test_postload_moment_corruption_cannot_pass():
    s=saved_state();m,o=model_optimizer();warm.restore(m,o,s,rng_state,{})
    o.state[m.actor[4].weight]['exp_avg'][0,300]=0
    with pytest.raises(ValueError,match='bytes differ'):warm.verify_restoration(m,o,s,rng_state())

def test_postload_learned_added_path_zeroing_cannot_pass():
    s=saved_state();m,o=model_optimizer();warm.restore(m,o,s,rng_state,{})
    with torch.no_grad():m.actor[4].weight[:,256:]=0
    with pytest.raises(ValueError,match='bytes differ'):warm.verify_restoration(m,o,s,rng_state())

def test_whole_optimizer_extra_group_metadata_preserved():
    s=saved_state();s['optimizer_state']['param_groups'][0]['qualification_tag']='synthetic-retain-me'
    m,o=model_optimizer();warm.restore(m,o,s,rng_state,{})
    assert o.param_groups[0]['qualification_tag']=='synthetic-retain-me'

def test_adapter_no_checkpoint_io_expansion_forward_or_update_calls():
    tree=ast.parse((Path(__file__).parent/'warm512_restore.py').read_text())
    for node in ast.walk(tree):
        if isinstance(node,ast.Call):
            name=node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else ''
            assert name not in {'load','save','open','read_bytes','read_text','expand_source','verify_width_restoration','forward','step','backward'}
