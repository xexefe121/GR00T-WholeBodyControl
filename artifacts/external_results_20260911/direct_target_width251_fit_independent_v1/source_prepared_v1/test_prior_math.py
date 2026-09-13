"""Synthetic arrays/graph only. Never load task data or execute any model."""
from pathlib import Path
import ast,copy
import numpy as np
import pytest
import torch
from audit_context_math import WIDTHS,source_context,endpoint_context,moments,rate,drift,sign_components
from audit_full_state_math import schedule
from audit_graph import audit_graph

def context_fixture():
    n=4;contract=dict(default_q=[0.]*23,kp=[2.]*23,training_effort=[4.]*23)
    a=dict(control=np.arange(250,254,dtype=np.int64),state=np.arange(n*52,dtype=np.float32).reshape(n,52),
        previous_action=np.full((n,23),.25,np.float32),expert_target=np.full((n,23),.125,np.float64))
    terms=(a['previous_action'],a['state'][:,49:52],a['state'][:,:23],a['state'][:,23:46],a['state'][:,46:49])
    for (name,width),term in zip(WIDTHS,terms):
        h=np.zeros((n,4,width),np.float32)
        for i in range(1,n):h[i,0]=term[i-1];h[i,1:]=h[i-1,:3]
        a['history_'+name]=h
    a['history']=np.concatenate([a['history_'+name].reshape(n,-1) for name,width in WIDTHS],axis=1)
    return a,contract

def test_exact_chronology_and_prior_order():
    a,c=context_fixture();context,n=source_context(a,c)
    assert n==3 and context.shape==(4,323)
    assert context[:,:23].tobytes()==a['previous_action'].tobytes()
    assert context[:,23:].tobytes()==a['history'].tobytes()

def test_coherently_shifted_named_and_flat_history_rejected():
    a,c=context_fixture()
    for name,width in WIDTHS:a['history_'+name][1:]=a['history_'+name][:-1].copy()
    a['history']=np.concatenate([a['history_'+name].reshape(4,-1) for name,width in WIDTHS],axis=1)
    with pytest.raises(AssertionError):source_context(a,c)

def test_nominal_prior_must_follow_applied_target():
    a,c=context_fixture();a['expert_target'][1,0]+=.01
    with pytest.raises(AssertionError):source_context(a,c)

def endpoint_fixture():
    a,c=context_fixture();start=np.array([0,2],np.int64)
    p=dict(center_index=start,incoming_history=a['history'][start].copy(),incoming_raw_prior=a['previous_action'][start].copy(),
        policy_applied_target=np.full((2,23),.5,np.float64),policy_actual_normalized_action=np.ones((2,23),np.float32),outgoing_raw_prior=np.full((2,23),9.,np.float32))
    for name,width in WIDTHS:p['advanced_history_'+name]=a['history_'+name][start+1].copy()
    p['advanced_history']=np.concatenate([p['advanced_history_'+name].reshape(2,-1) for name,width in WIDTHS],axis=1)
    return a,p,c

def test_endpoint_uses_applied_prior_and_once_advanced_history():
    a,p,c=endpoint_fixture();x=endpoint_context(a,p,c)
    assert np.all(x[:,:23]==1) and not np.array_equal(x[:,:23],p['outgoing_raw_prior'])
    assert x[:,23:].tobytes()==p['advanced_history'].tobytes()

def test_endpoint_double_shift_rejected():
    a,p,c=endpoint_fixture();p['advanced_history_actions'][:,1]=99
    with pytest.raises(AssertionError):endpoint_context(a,p,c)

def test_equal_cell_moments_differ_from_unbalanced_rows():
    from audit_math import phase_indices
    x=np.empty((9904,323),np.float32)
    for i,ids in enumerate(phase_indices()):x[ids]=i
    m,v,m32,s32=moments(x)
    assert np.all(m==7.) and np.allclose(v,np.var(np.arange(15,dtype=np.float64)))
    assert not np.isclose(x[:,0].mean(),m[0]) and m32.dtype==s32.dtype==np.float32
    _,v,_,std=moments(np.ones_like(x));assert np.all(v==0) and np.all(std==np.float32(.05))

def test_fixed_lower_schedule_and_prefix():
    assert rate(0)==1e-5 and rate(2999)==1e-6
    assert np.all(np.diff([rate(i) for i in range(3000)])<0)
    with pytest.raises(ValueError):rate(3000)
    c,a,_,_=schedule(2);c2,a2,_,_=schedule(3)
    assert c.tobytes()==c2[:2].tobytes() and a.tobytes()==a2[:2].tobytes()

def test_drift_widens_before_difference_and_uses_original_span():
    a=np.full((1,23),.123,np.float32);b=a.copy();b[0,0]=np.nextafter(b[0,0],np.float32(np.inf))
    span=np.full(23,3.1,np.float32);d=drift(a,b,span)
    assert d['max_preclip_error_rad']==float(abs(np.float64(a[0,0])-np.float64(b[0,0]))*np.float64(span[0])) and not d['byte_equal']

def test_odd_even_and_zero_baseline_keep_asymmetric_signs():
    error=np.empty((4,2,23),np.float64);error[:,0]=-2;error[:,1]=4
    wanted=np.empty_like(error);wanted[:,0]=-.5;wanted[:,1]=.25
    c=sign_components(error,wanted)
    assert c==dict(odd_response_MSE=9.,even_response_MSE=1.,zero_response_MSE=.15625)
    assert c['odd_response_MSE']+c['even_response_MSE']==np.square(error).mean()

def graph_fixture():
    import onnx
    from onnx import helper as h,numpy_helper as n,TensorProto as T
    actor={};arrays=[]
    for name,shape in [('0.weight',(512,1323)),('0.bias',(512,)),('2.weight',(512,512)),('2.bias',(512,)),('4.weight',(23,512)),('4.bias',(23,))]:actor[name]=torch.zeros(shape,dtype=torch.float32)
    checkpoint=dict(actor_state=actor,feature_mean=torch.zeros(1323),feature_std=torch.ones(1323))
    norm={k:checkpoint[k].numpy().copy() for k in ('feature_mean','feature_std')}
    arrays=[n.from_array(norm['feature_mean'].astype(np.float64),'mean'),n.from_array(norm['feature_std'].astype(np.float64),'std'),n.from_array(np.array(0.,np.float64),'zero'),n.from_array(np.array(1.,np.float64),'one')]
    nodes=[h.make_node('Cast',['features'],['features64'],to=T.DOUBLE),h.make_node('Sub',['features64','mean'],['center']),h.make_node('Div',['center','std'],['normalized'])]
    prior='normalized'
    for i,j in enumerate((0,2,4)):
        arrays += [n.from_array(actor[str(j)+'.weight'].numpy().astype(np.float64).T.copy(),'w'+str(i)),n.from_array(actor[str(j)+'.bias'].numpy().astype(np.float64),'b'+str(i))]
        nodes += [h.make_node('MatMul',[prior,'w'+str(i)],['m'+str(i)]),h.make_node('Add',['m'+str(i),'b'+str(i)],['a'+str(i)])]
        if i<2:
            nodes += [h.make_node('Min',['a'+str(i),'zero'],['negative'+str(i)]),h.make_node('Exp',['negative'+str(i)],['exp'+str(i)]),h.make_node('Sub',['exp'+str(i),'one'],['elu_negative'+str(i)]),h.make_node('Greater',['a'+str(i),'zero'],['positive'+str(i)]),h.make_node('Where',['positive'+str(i),'a'+str(i),'elu_negative'+str(i)],['e'+str(i)])]
            prior='e'+str(i)
        else:prior='a'+str(i)
    nodes.append(h.make_node('Cast',[prior],['normalized_target'],to=T.FLOAT))
    graph=h.make_model(h.make_graph(nodes,'synthetic',[h.make_tensor_value_info('features',T.FLOAT,[None,1323])],[h.make_tensor_value_info('normalized_target',T.FLOAT,[None,23])],arrays),opset_imports=[h.make_opsetid('',17)]);graph.ir_version=8
    return graph,checkpoint,norm

def test1323_graph_exact_saved_promotions_no_inference():
    g,c,n=graph_fixture();assert audit_graph(g,c,n)['graph_nodes']==20

@pytest.mark.parametrize('mutation',['old_width','wrong_context_mean','wrong_weight','wrong_output_type','wrong_wire'])
def test_graph_corruption_rejected(mutation):
    from onnx import numpy_helper,TensorProto
    g,c,n=graph_fixture()
    if mutation=='old_width':g.graph.input[0].type.tensor_type.shape.dim[1].dim_value=1000
    elif mutation=='wrong_output_type':g.graph.node[-1].attribute[0].i=TensorProto.DOUBLE
    elif mutation=='wrong_wire':g.graph.node[3].input[0]='features64'
    else:
        name='mean' if mutation=='wrong_context_mean' else 'w0'
        item=next(a for a in g.graph.initializer if a.name==name);x=numpy_helper.to_array(item).copy();x.flat[-1]=1;item.CopyFrom(numpy_helper.from_array(x,name))
    with pytest.raises(AssertionError):audit_graph(g,c,n)

def test_audit_source_never_imports_or_calls_task_models():
    folder=Path(__file__).parent
    for file in ['audit_saved_warm.py','audit_balanced_math.py','audit_release.py','audit_context_math.py','audit_graph.py','audit_full_state_math.py','audit_math.py','audit_restoration.py']:
        tree=ast.parse((folder/file).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):assert not any(a.name.startswith(('onnxruntime','mujoco','context_model','context_data','direct_data','full_state_data')) for a in node.names)
            if isinstance(node,ast.ImportFrom):assert not (node.module or '').startswith(('onnxruntime','mujoco','context_model','context_data','direct_data','full_state_data'))
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):assert node.func.attr not in ('backward','step','InferenceSession','mj_step')
