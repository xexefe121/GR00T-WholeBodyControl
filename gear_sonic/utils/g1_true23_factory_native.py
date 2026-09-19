"""Exact trainable reconstruction of the verified native23 factory network."""
from pathlib import Path
import numpy as np
import torch
from torch import nn


def dense_stack(weights,name,indices,*,tanh):
    layers=[]
    for i in indices:
        key=f'{name}.{name}.{i}'
        w,b=weights[key+'.weight'],weights[key+'.bias']
        layer=nn.Linear(w.shape[1],w.shape[0])
        with torch.no_grad():layer.weight.copy_(torch.from_numpy(w));layer.bias.copy_(torch.from_numpy(b))
        layers.append(layer)
        if i!=indices[-1]:layers.append(nn.ELU())
    if tanh:layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class Native23FactoryNetwork(nn.Module):
    def __init__(self,weights_path:Path):
        super().__init__()
        with np.load(weights_path,allow_pickle=False) as z:w={k:z[k].copy() for k in z.files}
        self.memory=dense_stack(w,'mem_encoder',[0,2,4],tanh=True)
        self.estimator=dense_stack(w,'state_estimator',[0,2,4],tanh=True)
        self.actor=dense_stack(w,'low_level_net',[0,2,4,6],tanh=True)

    def forward(self,obs):
        encoded=self.memory(obs[:,:,:75].reshape(-1,375))
        phase=obs[:,-1,75:76]
        velocity=self.estimator(encoded)
        return self.actor(torch.cat((encoded,phase,velocity),-1))


class NativeLegFactoryNetwork(nn.Module):
    """Exact g1_b_l_ankle_track: memory210->64, commands8, raw leg targets12."""
    def __init__(self,weights_path):
        super().__init__()
        with np.load(weights_path,allow_pickle=False) as z:w={k.lstrip('.'):z[k].copy() for k in z.files}
        self.memory=dense_stack(w,'mem_encoder',[0,2,4],tanh=True)
        layers=list(dense_stack(w,'low_level_net',[0,2,4],tanh=False).children())
        final=nn.Linear(32,12)
        with torch.no_grad():
            final.weight.copy_(torch.from_numpy(w['act__matmul_converted.weight']))
            final.bias.copy_(torch.from_numpy(w['act__matmul_converted.bias']))
        self.actor=nn.Sequential(*layers,nn.ELU(),final)

    def forward(self,p_obs,cmd):
        encoded=self.memory(p_obs.reshape(-1,210))
        return self.actor(torch.cat((encoded,cmd),-1))
