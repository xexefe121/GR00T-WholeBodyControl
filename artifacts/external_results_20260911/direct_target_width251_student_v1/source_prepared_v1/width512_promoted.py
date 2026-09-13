"""Immutable float32 weights promoted to float64 execution; no fitting API."""
import numpy as np
import torch
from torch.nn import functional as F

class PromotedDirect(torch.nn.Module):
    def __init__(self,mean,std,weights,biases):
        super().__init__()
        arrays=[np.asarray(mean),np.asarray(std)]+[np.asarray(v) for pair in zip(weights,biases) for v in pair]
        if any(v.dtype!=np.float32 or not np.isfinite(v).all() for v in arrays):raise ValueError('All source values must be finite original float32')
        if len(weights)!=3 or len(biases)!=3 or mean.shape!=std.shape or len(mean.shape)!=1 or np.any(std<=0):raise ValueError('Normalization/layer schema')
        size=len(mean)
        for w,b in zip(weights,biases):
            if w.ndim!=2 or w.shape[1]!=size or b.shape!=(w.shape[0],):raise ValueError('Layer dimensions')
            size=w.shape[0]
        self.register_buffer('feature_mean',torch.from_numpy(mean.copy()).to(torch.float64))
        self.register_buffer('feature_std',torch.from_numpy(std.copy()).to(torch.float64))
        for i,(w,b) in enumerate(zip(weights,biases)):
            self.register_buffer('w'+str(i),torch.from_numpy(w.copy()).to(torch.float64))
            self.register_buffer('b'+str(i),torch.from_numpy(b.copy()).to(torch.float64))
    def forward(self,features):
        if features.dtype!=torch.float32:raise ValueError('Public feature input must stay float32')
        value=(features.to(torch.float64)-self.feature_mean)/self.feature_std
        for i in range(3):
            value=F.linear(value,getattr(self,'w'+str(i)),getattr(self,'b'+str(i)))
            if i<2:value=F.elu(value,alpha=1.)
        return value.to(torch.float32)

def from_checkpoint(saved):
    if saved['kind']!='direct_absolute_native23_target' or saved['ordinary_final_step']!=81000:raise ValueError('Exact proposed ordinary81000 subject required')
    actor=saved['actor_state'];expected=['0.weight','0.bias','2.weight','2.bias','4.weight','4.bias']
    if set(actor)!=set(expected):raise ValueError('Unexpected actor parameters')
    weights=[actor[str(i)+'.weight'].cpu().numpy() for i in (0,2,4)]
    biases=[actor[str(i)+'.bias'].cpu().numpy() for i in (0,2,4)]
    if [w.shape for w in weights]!=[(512,1323),(512,512),(23,512)]:raise ValueError('Causal context architecture changed')
    return PromotedDirect(saved['feature_mean'].cpu().numpy(),saved['feature_std'].cpu().numpy(),weights,biases)

def export_onnx(model,path):
    """No tracing/forward call; alpha1 ELU decomposed because CPU ORT has no double Elu kernel."""
    import onnx
    from onnx import helper,numpy_helper,TensorProto
    arrays=[numpy_helper.from_array(model.feature_mean.cpu().numpy().copy(),'mean'),
            numpy_helper.from_array(model.feature_std.cpu().numpy().copy(),'std'),
            numpy_helper.from_array(np.array(0.,np.float64),'zero'),numpy_helper.from_array(np.array(1.,np.float64),'one')]
    nodes=[helper.make_node('Cast',['features'],['features64'],to=TensorProto.DOUBLE),
        helper.make_node('Sub',['features64','mean'],['center']),helper.make_node('Div',['center','std'],['normalized'])]
    value='normalized'
    for i in range(3):
        arrays.extend([numpy_helper.from_array(getattr(model,'w'+str(i)).cpu().numpy().T.copy(),'w'+str(i)),numpy_helper.from_array(getattr(model,'b'+str(i)).cpu().numpy().copy(),'b'+str(i))])
        nodes.extend([helper.make_node('MatMul',[value,'w'+str(i)],['m'+str(i)]),helper.make_node('Add',['m'+str(i),'b'+str(i)],['a'+str(i)])]);value='a'+str(i)
        if i<2:
            # Exp is evaluated only on min(x,0), preventing overflow on the unused positive branch.
            nodes.extend([helper.make_node('Min',[value,'zero'],['negative'+str(i)]),helper.make_node('Exp',['negative'+str(i)],['exp'+str(i)]),
                helper.make_node('Sub',['exp'+str(i),'one'],['elu_negative'+str(i)]),helper.make_node('Greater',[value,'zero'],['positive'+str(i)]),
                helper.make_node('Where',['positive'+str(i),value,'elu_negative'+str(i)],['e'+str(i)])]);value='e'+str(i)
    nodes.append(helper.make_node('Cast',[value],['normalized_target'],to=TensorProto.FLOAT))
    graph=helper.make_graph(nodes,'same_81000_width512_context_weights_fp64_execution',[helper.make_tensor_value_info('features',TensorProto.FLOAT,[None,len(model.feature_mean)])],
        [helper.make_tensor_value_info('normalized_target',TensorProto.FLOAT,[None,len(model.b2)])],arrays)
    exported=helper.make_model(graph,opset_imports=[helper.make_opsetid('',17)]);exported.ir_version=8
    onnx.checker.check_model(exported);onnx.save(exported,str(path))
