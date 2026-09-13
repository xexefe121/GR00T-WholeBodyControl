"""Synthetic CPU models only. Never reads a task checkpoint or task corpus."""
import copy
import numpy as np
import pytest
import torch
from context_model import ContextTarget
from response_restore import check_warm_source,verify_warm_restoration
from response_contract import COEFFICIENT,cosine_rate,BUDGETS
from response_diagnostics import add_balanced_metrics,save_balanced_prefix
from balance_contract import GROUP_WEIGHTS,ZERO_RESPONSE_ENERGIES,MEAN_ZERO_RESPONSE_ENERGY
from restoration_support import exact_saved
from training_support import cpu_tree
from train_response_balanced import initial_drift
from response_promoted import from_checkpoint,export_onnx

torch.set_num_threads(1)

@pytest.fixture
def warm():
    torch.manual_seed(517)
    model=ContextTarget(np.zeros(1323,np.float32),np.ones(1323,np.float32))
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-6,weight_decay=1e-5,foreach=False,fused=False)
    inputs=torch.randn(3,1323)
    optimizer.zero_grad(set_to_none=True);model(inputs).square().mean().backward();optimizer.step()
    for state in optimizer.state.values():state['step'].fill_(3000)
    saved=dict(kind='direct_absolute_native23_target',condition='causal',ordinary_final_step=68000,
        optimizer_step=3000,context_blinded=False,full_state_coefficient=COEFFICIENT,
        context_order='previous_action23_then_incoming_history300',actor_state=cpu_tree(model.actor.state_dict()),
        optimizer_state=cpu_tree(optimizer.state_dict()),feature_mean=model.feature_mean.clone(),
        feature_std=model.feature_std.clone(),rng={'synthetic_cpu':torch.get_rng_state().clone()})
    return model,optimizer,saved,inputs

def restored(saved):
    model=ContextTarget(saved['feature_mean'].numpy(),saved['feature_std'].numpy())
    model.actor.load_state_dict(saved['actor_state'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-5,foreach=False,fused=False)
    optimizer.load_state_dict(copy.deepcopy(saved['optimizer_state']))
    return model,optimizer

def test_warm_restore_preserves_learned_context_and_all_moments(warm):
    _,_,saved,_=warm;model,opt=restored(saved)
    record=verify_warm_restoration(model,opt,saved,copy.deepcopy(saved['rng']))
    assert record['optimizer_start_step']==3000 and record['fresh_optimizer'] is False
    assert torch.count_nonzero(model.actor[0].weight[:,1000:])>0
    assert opt.param_groups[0]['lr']==1e-6

@pytest.mark.parametrize('mutation',['actor','context','moment','step','norm','RNG','group'])
def test_actual_restoration_corruptions_rejected(warm,mutation):
    _,_,saved,_=warm;model,opt=restored(saved);rng=copy.deepcopy(saved['rng'])
    with torch.no_grad():
        if mutation=='actor':model.actor[2].weight[0,0]+=1
        if mutation=='context':model.actor[0].weight[:,1000:].zero_()
        if mutation=='moment':next(iter(opt.state.values()))['exp_avg'].flatten()[0]+=1
        if mutation=='step':next(iter(opt.state.values()))['step'].fill_(0)
        if mutation=='norm':model.feature_mean[1000]+=1
        if mutation=='RNG':rng['synthetic_cpu'][0]^=1
        if mutation=='group':opt.param_groups[0]['lr']=1e-5
    with pytest.raises(ValueError):verify_warm_restoration(model,opt,saved,rng)

def test_warm_next_step_matches_original_moment_continuation(warm):
    original,oldopt,saved,inputs=warm;model,opt=restored(saved)
    for actor,optimizer in [(original,oldopt),(model,opt)]:
        optimizer.param_groups[0]['lr']=cosine_rate(0)
        optimizer.zero_grad(set_to_none=True);actor(inputs).square().mean().backward();optimizer.step()
    exact_saved(model.actor.state_dict(),original.actor.state_dict())
    exact_saved(opt.state_dict(),oldopt.state_dict())
    assert all(float(state['step'])==3001 for state in opt.state.values())

@pytest.mark.parametrize('mutation',['fractional_step','duplicate_id','actor_order','bad_flags'])
def test_invalid_source_contract_rejected(warm,mutation):
    saved=copy.deepcopy(warm[2])
    if mutation=='fractional_step':next(iter(saved['optimizer_state']['state'].values()))['step'].fill_(3000.5)
    if mutation=='duplicate_id':saved['optimizer_state']['param_groups'][0]['params'][1]=0
    if mutation=='actor_order':saved['actor_state']=dict(reversed(list(saved['actor_state'].items())))
    if mutation=='bad_flags':saved['optimizer_state']['param_groups'][0]['capturable']=True
    with pytest.raises(ValueError):check_warm_source(saved)

def metric_fixture():
    cells=[]
    for d in range(3):
        for p in range(3):
            for g in range(6):
                value=(d*18+p*6+g+1)*.001
                cells.append(dict(dataset=d,phase=p,tangent_group=g,response_MSE=value,
                    first24_response_MSE=value/2,odd_response_MSE=value*.8,
                    even_response_MSE=value*.2,zero_response_MSE=ZERO_RESPONSE_ENERGIES[g]))
    return dict(full_state_cells=cells,nominal_objective=.2,physical_objective=.3,
                full_state_objective=float(np.mean([c['response_MSE'] for c in cells])))

def test_weighted_diagnostics_preserve_original_fields_and_teacher_baseline():
    source=metric_fixture();old=copy.deepcopy(source);result=add_balanced_metrics(source)
    for key,value in old.items():assert result[key]==value
    expected=np.mean([c['response_MSE']*GROUP_WEIGHTS[c['tangent_group']] for c in old['full_state_cells']])
    assert result['balanced_full_state_objective']==expected
    assert np.isclose(result['balanced_full_state_zero_response_MSE'],MEAN_ZERO_RESPONSE_ENERGY,rtol=1e-14,atol=0)
    assert result['weighted_objective']==.2+COEFFICIENT*old['full_state_objective']+.3
    assert result['balanced_weighted_objective']==.2+COEFFICIENT*expected+.3

def test_wrong_metric_cell_order_rejected():
    value=metric_fixture();value['full_state_cells'][0],value['full_state_cells'][1]=value['full_state_cells'][1],value['full_state_cells'][0]
    with pytest.raises(ValueError):add_balanced_metrics(value)

def test_fixed_lr_and_single_condition_budgets():
    assert cosine_rate(0)==1e-5 and cosine_rate(2999)==1e-6
    assert all(cosine_rate(i)>=cosine_rate(i+1) for i in range(2999))
    assert BUDGETS['training_forward_rows']==3000*(9904+1728+3054)
    assert BUDGETS['training_forward_calls']==9000 and BUDGETS['training_updates']==3000

def test_failure_prefix_keeps_original_and_balanced_records(tmp_path):
    save_balanced_prefix(tmp_path,[[1,2,3,4,5,6]],[list(range(15))],[list(range(54))],
        [list(range(9))],[[7,8]],[list(np.arange(54)*3)])
    assert np.load(tmp_path/'training_progress.npy').shape==(1,6)
    assert np.load(tmp_path/'original_objectives.npy').tolist()==[[7,8]]
    assert np.load(tmp_path/'full_state_cell_losses.npy')[0,7]==7
    assert np.load(tmp_path/'balanced_full_state_cell_losses.npy')[0,7]==21

def test_initial_drift_gate_allows_disclosed_nonbyte_difference(tmp_path):
    outputs={};paths={}
    for name in ('nominal','full_state','physical'):
        before=np.ones((3,23),np.float32);after=before.copy();after[0,0]+=np.float32(1e-6)
        path=tmp_path/(name+'.npy');np.save(path,before);paths[name]=str(path);outputs[name]=after
    result=initial_drift(outputs,{'restoration_predictions':paths},{'span':np.ones(23,np.float32)})
    assert result['passed'] and not result['byte_gate_required']
    assert all(not x['byte_equal'] for x in result['corpora'].values())
    outputs['physical'][0,0]+=np.float32(1e-3)
    assert not initial_drift(outputs,{'restoration_predictions':paths},{'span':np.ones(23,np.float32)})['passed']

def test_export_ordinary71000_and_public_dtypes_without_trace(warm,tmp_path):
    import onnx
    saved=copy.deepcopy(warm[2]);saved['ordinary_final_step']=71000
    promoted=from_checkpoint(saved);path=tmp_path/'synthetic.onnx';export_onnx(promoted,path)
    graph=onnx.load(path).graph
    assert graph.input[0].type.tensor_type.elem_type==onnx.TensorProto.FLOAT
    assert graph.output[0].type.tensor_type.elem_type==onnx.TensorProto.FLOAT
    assert graph.input[0].type.tensor_type.shape.dim[1].dim_value==1323
    assert len(graph.node)==20
    arrays={a.name:onnx.numpy_helper.to_array(a) for a in graph.initializer}
    np.testing.assert_array_equal(arrays['w0'],saved['actor_state']['0.weight'].numpy().T.astype(np.float64))
    assert all(a.dtype==np.float64 for a in arrays.values())
