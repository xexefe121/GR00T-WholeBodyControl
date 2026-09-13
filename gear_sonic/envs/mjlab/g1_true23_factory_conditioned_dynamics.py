"""Native23 factory balance plus full received-body command conditioning."""
from pathlib import Path
import numpy as np
import torch
import yaml
from gear_sonic.envs.mjlab.g1_true23_factory_dynamics import FactoryNative23Env
from gear_sonic.envs.mjlab.g1_true23_causal_dynamics import features_torch,quaternion_matrix
from gear_sonic.utils.g1_true23_factory_policy import IDS


class NativeFactoryConditionedEnv(FactoryNative23Env):
    def __init__(self,bundle,bank,config,count=128,device='cuda:0'):
        super().__init__(bundle,bank,config,count,device)
        self.native_cfg=yaml.safe_load(Path(config).read_text())
        self.action_scale=self.old_scale.clone()  # Existing received feature history convention.
        self.native_history=torch.zeros((count,4,76),device=device)
        self.native_previous=torch.zeros((count,23),device=device)
        with np.load(Path(bank)/'bfm_reference_inputs_v1.npz',allow_pickle=False) as z:
            if not bool(z['task_closure_stand']):raise ValueError('native conditioning needs received geometric standing closure')
            length=self.states.shape[1]
            alphas=[np.pad(z[f'alpha_{i}'],(0,length-len(z[f'alpha_{i}'])),mode='edge') for i in range(len(self.meta['clips']))]
        self.native_alpha=torch.tensor(np.stack(alphas),device=device)
        self.native_ready=True
        self.reset(torch.arange(count,device=device))

    def native_prop(self):
        cfg=self.native_cfg
        velocity=self.v[:,6:].clone();velocity[:,[4,5,10,11]]=0
        return torch.cat((self.v[:,3:6]*cfg['observation_scale_ang_vel'],-quaternion_matrix(self.q[:,3:7])[:,2]*cfg['observation_scale_proj_grav'],
            (self.q[:,7:]-self.factory_default)*cfg['observation_scale_dof_pos'],velocity*cfg['observation_scale_dof_vel'],
            self.native_previous*cfg['observation_scale_actions'],self.q.new_zeros((self.count,1))),-1).clamp(-cfg['observation_clip'],cfg['observation_clip'])

    def observe(self):
        if not getattr(self,'native_ready',False):return super().observe()
        prop=self.native_prop()
        native=torch.cat((self.native_history,prop[:,None]),1).flatten(1)
        features=features_torch(self.q,self.v,self.references,self.frames,self.clips,self.default,
            self.prior,self.history_vector(),self.lengths)
        index=torch.minimum(self.frames,self.lengths[self.clips]-1)
        return torch.cat((native,features,self.native_alpha[self.clips,index,None]),-1)

    def reset(self,ids,canonical=False):
        super().reset(ids,canonical)
        if not getattr(self,'native_ready',False) or not len(ids):return
        # Actual expert previous target transformed to native factory units.
        self.native_previous[ids]=self.factory_prior[ids][:,IDS]*.25
        self.native_history[ids]=self.native_prop()[ids,None]

    def step(self,targets):
        prop=self.native_prop().clone()
        self.native_history[:,:-1]=self.native_history[:,1:].clone();self.native_history[:,-1]=prop
        self.native_previous[:]=targets-self.factory_default
        return super().step(targets)
