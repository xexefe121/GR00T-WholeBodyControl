"""Same six-parameter MLP, with323 zero-initialized causal context columns."""
import numpy as np
import torch
from restoration_support import exact_saved

class ContextTarget(torch.nn.Module):
    def __init__(self,mean,std):
        super().__init__()
        if mean.shape!=(1323,) or std.shape!=(1323,) or mean.dtype!=np.float32 or std.dtype!=np.float32 or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std<=0):raise ValueError('1323 normalization schema.')
        self.register_buffer('feature_mean',torch.from_numpy(mean.copy()))
        self.register_buffer('feature_std',torch.from_numpy(std.copy()))
        self.actor=torch.nn.Sequential(torch.nn.Linear(1323,256),torch.nn.ELU(),torch.nn.Linear(256,256),torch.nn.ELU(),torch.nn.Linear(256,23))
    def forward(self,features):
        if features.dtype!=torch.float32 or features.ndim!=2 or features.shape[1]!=1323:raise ValueError('Public context feature schema.')
        normalized=(features-self.feature_mean)/self.feature_std
        first=self.actor[0]
        # Keep the original1000 contraction dimensions/layout at initialization.
        # Contiguous copies are differentiable views of the same six parameters.
        original=torch.nn.functional.linear(normalized[:,:1000].contiguous(),first.weight[:,:1000].contiguous(),first.bias)
        context=torch.nn.functional.linear(normalized[:,1000:].contiguous(),first.weight[:,1000:].contiguous(),None)
        value=original+context
        for index in range(1,len(self.actor)):value=self.actor[index](value)
        return value

def expand_actor(original):
    expected={'0.weight':(256,1000),'0.bias':(256,),'2.weight':(256,256),'2.bias':(256,),'4.weight':(23,256),'4.bias':(23,)}
    if set(original)!=set(expected):raise ValueError('Original six actor tensors required.')
    for key,shape in expected.items():
        if tuple(original[key].shape)!=shape or original[key].dtype!=torch.float32 or not bool(torch.isfinite(original[key]).all()):raise ValueError('Source actor schema: '+key)
    expanded={key:value.detach().cpu().clone() for key,value in original.items()}
    expanded['0.weight']=torch.cat((expanded['0.weight'],torch.zeros(256,323,dtype=torch.float32)),dim=1)
    for key,value in original.items():exact_saved(expanded[key][:,:1000] if key=='0.weight' else expanded[key],value,'expanded '+key)
    if torch.count_nonzero(expanded['0.weight'][:,1000:])!=0:raise ValueError('New context columns must start exactly zero.')
    return expanded

def assert_blinded_columns(actor):
    if torch.count_nonzero(actor['0.weight'][:,1000:])!=0:raise ValueError('Blinded new columns changed despite normalized-zero input.')
