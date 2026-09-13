"""Synthetic parameter/graph checks only: no Torch forward or ORT evaluation."""
import ast,copy
from pathlib import Path
import numpy as np
import onnx
import pytest
import torch
from onnx import numpy_helper,TensorProto
from width512_promoted import PromotedDirect,from_checkpoint,export_onnx

torch.set_num_threads(1)
SHAPES=((512,1323),(512,),(512,512),(512,),(23,512),(23,))
NAMES=('0.weight','0.bias','2.weight','2.bias','4.weight','4.bias')

def same(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def synthetic():
    arrays=[]
    for shape in SHAPES:
        a=np.linspace(-.0625,.0625,num=int(np.prod(shape)),dtype=np.float32).reshape(shape)
        a.flat[0]=-0.;a.flat[1]=+0.;arrays.append(a)
    return dict(kind='direct_absolute_native23_target',condition='causal',ordinary_final_step=81000,optimizer_step=16000,
        actor_state={n:torch.from_numpy(v.copy()) for n,v in zip(NAMES,arrays)},
        feature_mean=torch.from_numpy(np.linspace(-.1,.1,1323,dtype=np.float32)),
        feature_std=torch.from_numpy(np.linspace(.05,2,1323,dtype=np.float32)))

def expected_nodes():
    nodes=[('Cast',['features'],['features64']),('Sub',['features64','mean'],['center']),('Div',['center','std'],['normalized'])]
    value='normalized'
    for i in range(3):
        nodes.extend([('MatMul',[value,'w'+str(i)],['m'+str(i)]),('Add',['m'+str(i),'b'+str(i)],['a'+str(i)])]);value='a'+str(i)
        if i<2:
            nodes.extend([('Min',[value,'zero'],['negative'+str(i)]),('Exp',['negative'+str(i)],['exp'+str(i)]),
                ('Sub',['exp'+str(i),'one'],['elu_negative'+str(i)]),('Greater',[value,'zero'],['positive'+str(i)]),
                ('Where',['positive'+str(i),value,'elu_negative'+str(i)],['e'+str(i)])]);value='e'+str(i)
    nodes.append(('Cast',[value],['normalized_target']))
    return nodes

def test_parameter_and_normalization_promotion_roundtrip():
    saved=synthetic();before=copy.deepcopy(saved);rng=torch.get_rng_state().clone();model=from_checkpoint(saved)
    assert torch.equal(rng,torch.get_rng_state())
    assert not list(model.parameters())
    for key in ('feature_mean','feature_std'):
        actual=getattr(model,key).numpy();original=saved[key].numpy()
        assert same(actual,original.astype(np.float64)) and same(actual.astype(np.float32),original)
    for i,key in enumerate((0,2,4)):
        for letter,field in [('w','weight'),('b','bias')]:
            original=saved['actor_state'][str(key)+'.'+field].numpy();actual=getattr(model,letter+str(i)).numpy()
            assert actual.dtype==np.float64 and same(actual,original.astype(np.float64))
            assert same(actual.astype(np.float32),original)
    for n,v in saved['actor_state'].items():assert same(v.numpy(),before['actor_state'][n].numpy())

def test_buffers_are_owned():
    saved=synthetic();model=from_checkpoint(saved);before={n:v.numpy().copy() for n,v in model.named_buffers()}
    for value in saved['actor_state'].values():value.fill_(5)
    saved['feature_mean'].fill_(9);saved['feature_std'].fill_(3)
    for n,value in model.named_buffers():assert same(value.numpy(),before[n])

def test_exact_manual_graph_without_forward(tmp_path,monkeypatch):
    saved=synthetic();model=from_checkpoint(saved)
    def forbidden(*args,**kwargs):raise AssertionError('No synthetic or actual forward allowed in graph export checks')
    monkeypatch.setattr(model,'forward',forbidden)
    monkeypatch.setattr(torch.nn.functional,'linear',forbidden)
    target=tmp_path/'synthetic_width512.onnx';export_onnx(model,target)
    graph=onnx.load(str(target));onnx.checker.check_model(graph,full_check=True)
    assert graph.ir_version==8 and [(v.domain,v.version) for v in graph.opset_import]==[('',17)]
    assert graph.graph.name=='same_81000_width512_context_weights_fp64_execution'
    assert [(v.op_type,list(v.input),list(v.output)) for v in graph.graph.node]==expected_nodes()
    assert len(graph.graph.node)==20
    assert [v.name for v in graph.graph.input]==['features'] and [v.name for v in graph.graph.output]==['normalized_target']
    for value,width in [(graph.graph.input[0],1323),(graph.graph.output[0],23)]:
        t=value.type.tensor_type;assert t.elem_type==TensorProto.FLOAT
        assert len(t.shape.dim)==2 and not t.shape.dim[0].HasField('dim_value') and t.shape.dim[1].dim_value==width
    initializers={v.name:numpy_helper.to_array(v) for v in graph.graph.initializer}
    assert set(initializers)=={'mean','std','zero','one','w0','b0','w1','b1','w2','b2'}
    for i in range(3):
        assert same(initializers['w'+str(i)],getattr(model,'w'+str(i)).numpy().T.copy())
        assert same(initializers['b'+str(i)],getattr(model,'b'+str(i)).numpy())
    assert same(initializers['mean'],model.feature_mean.numpy()) and same(initializers['std'],model.feature_std.numpy())
    assert same(initializers['zero'],np.array(0.,np.float64)) and same(initializers['one'],np.array(1.,np.float64))
    assert all(v.data_type==TensorProto.DOUBLE and v.data_location==TensorProto.DEFAULT for v in graph.graph.initializer)
    for index,node in enumerate(graph.graph.node):
        attrs={a.name:onnx.helper.get_attribute_value(a) for a in node.attribute}
        assert attrs==({'to':TensorProto.DOUBLE} if index==0 else {'to':TensorProto.FLOAT} if index==19 else {})
        assert node.domain==''
    # Check shape inference without executing the model.
    inferred=onnx.shape_inference.infer_shapes(graph,strict_mode=True)
    inferred_types={v.name:v.type.tensor_type.elem_type for v in inferred.graph.value_info}
    assert inferred_types['features64']==TensorProto.DOUBLE and inferred_types['a2']==TensorProto.DOUBLE
    assert inferred_types['positive0']==inferred_types['positive1']==TensorProto.BOOL

@pytest.mark.parametrize('step',[71000,68000,81001,None])
def test_wrong_future_endpoint_rejected(step):
    saved=synthetic();saved['ordinary_final_step']=step
    with pytest.raises(ValueError):from_checkpoint(saved)

@pytest.mark.parametrize('case',['kind','extra_tensor','missing_tensor','old_width','wrong_input','wrong_output','norm_length','std_zero','std_negative','nan_mean','infinite_weight','float64_weight'])
def test_invalid_source_schema_rejected(case):
    saved=synthetic()
    if case=='kind':saved['kind']='residual_head'
    elif case=='extra_tensor':saved['actor_state']['extra']=torch.zeros(1)
    elif case=='missing_tensor':saved['actor_state'].pop('4.bias')
    elif case=='old_width':saved['actor_state']['0.weight']=torch.zeros((256,1323))
    elif case=='wrong_input':saved['actor_state']['0.weight']=torch.zeros((512,1322))
    elif case=='wrong_output':saved['actor_state']['4.weight']=torch.zeros((24,512))
    elif case=='norm_length':saved['feature_mean']=torch.zeros(1322)
    elif case=='std_zero':saved['feature_std'][0]=0
    elif case=='std_negative':saved['feature_std'][0]=-1
    elif case=='nan_mean':saved['feature_mean'][0]=float('nan')
    elif case=='infinite_weight':saved['actor_state']['2.weight'][0,0]=float('inf')
    elif case=='float64_weight':saved['actor_state']['2.weight']=saved['actor_state']['2.weight'].double()
    with pytest.raises(ValueError):from_checkpoint(saved)

def test_exporter_contains_no_ORT_tracing_or_checkpoint_read():
    p=Path(__file__).with_name('width512_promoted.py');tree=ast.parse(p.read_text())
    imports=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):imports.extend(v.name for v in node.names)
        if isinstance(node,ast.ImportFrom):imports.append(node.module)
    assert not any('onnxruntime' in name for name in imports)
    text=p.read_text();assert 'torch.load' not in text and 'torch.onnx.export' not in text
    export=next(v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name=='export_onnx')
    calls=[ast.unparse(v.func) for v in ast.walk(export) if isinstance(v,ast.Call)]
    assert 'model' not in calls and 'model.forward' not in calls and 'F.linear' not in calls
