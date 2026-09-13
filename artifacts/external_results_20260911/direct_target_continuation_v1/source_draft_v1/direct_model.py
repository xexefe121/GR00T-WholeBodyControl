"""Fresh MLP; ONNX returns normalized absolute targets, no BFM or span node."""
import numpy as np
import torch

class DirectTarget(torch.nn.Module):
    def __init__(self,mean,std):
        super().__init__()
        self.register_buffer('feature_mean',torch.as_tensor(mean,dtype=torch.float32).clone())
        self.register_buffer('feature_std',torch.as_tensor(std,dtype=torch.float32).clone())
        if self.feature_mean.shape!=(1000,) or self.feature_std.shape!=(1000,):raise ValueError('Normalization shape.')
        self.actor=torch.nn.Sequential(torch.nn.Linear(1000,256),torch.nn.ELU(),torch.nn.Linear(256,256),torch.nn.ELU(),torch.nn.Linear(256,23))
    def forward(self,features):return self.actor((features-self.feature_mean)/self.feature_std)

def export_onnx(model,path):
    import onnx
    from onnx import helper,numpy_helper,TensorProto
    arrays=[numpy_helper.from_array(model.feature_mean.detach().cpu().numpy().copy(),'mean'),
            numpy_helper.from_array(model.feature_std.detach().cpu().numpy().copy(),'std')]
    nodes=[helper.make_node('Sub',['features','mean'],['center']),helper.make_node('Div',['center','std'],['normalized'])]
    value='normalized'
    for n,index in enumerate((0,2,4)):
        layer=model.actor[index]
        arrays.extend([numpy_helper.from_array(layer.weight.detach().cpu().numpy().T.copy(),f'w{n}'),numpy_helper.from_array(layer.bias.detach().cpu().numpy().copy(),f'b{n}')])
        output='normalized_target' if n==2 else f'a{n}'
        nodes.extend([helper.make_node('MatMul',[value,f'w{n}'],[f'm{n}']),helper.make_node('Add',[f'm{n}',f'b{n}'],[output])]);value=output
        if n<2:nodes.append(helper.make_node('Elu',[value],[f'e{n}'],alpha=1.));value=f'e{n}'
    graph=helper.make_graph(nodes,'direct_native23_absolute_target',[helper.make_tensor_value_info('features',TensorProto.FLOAT,[None,1000])],
        [helper.make_tensor_value_info('normalized_target',TensorProto.FLOAT,[None,23])],arrays)
    exported=helper.make_model(graph,opset_imports=[helper.make_opsetid('',17)]);exported.ir_version=8
    onnx.checker.check_model(exported);onnx.save(exported,str(path))
