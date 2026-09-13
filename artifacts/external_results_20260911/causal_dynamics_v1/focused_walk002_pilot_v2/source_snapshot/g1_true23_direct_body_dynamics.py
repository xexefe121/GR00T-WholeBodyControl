"""Full current-body goal encoding with the same native dynamics and resets."""
import numpy as np
import torch
from gear_sonic.envs.mjlab.g1_true23_single_policy_dynamics import SinglePolicyNative23Env


class DirectBodyNative23Env(SinglePolicyNative23Env):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        with np.load(self.bank/'bfm_reference_inputs_v1.npz') as z:
            maximum=int(self.lengths.max());rows=[];alphas=[]
            for i in range(len(self.meta['clips'])):
                body=z[f'body_{i}'];alpha=z[f'alpha_{i}']
                rows.append(np.concatenate((body,np.repeat(body[-1:],maximum-len(body),0))))
                alphas.append(np.pad(alpha,(0,maximum-len(alpha)),mode='edge'))
        self.body_inputs=torch.tensor(np.stack(rows),device=self.device)
        self.reference_blend=torch.tensor(np.stack(alphas),device=self.device)
        clip=self.expert_resets[:,0].long();frame=self.expert_resets[:,1].long()
        frame=torch.minimum(frame,self.lengths[clip]-1)
        self.expert_features=torch.cat((self.expert_features,self.body_inputs[clip,frame],self.reference_blend[clip,frame,None]),-1)
        self.reset(torch.arange(self.count,device=self.device),canonical=True)

    def reset(self,ids,canonical=False):
        super().reset(ids,canonical)
        if hasattr(self,'reference_blend'):
            clip=self.clips[ids];frame=torch.minimum(self.frames[ids],self.lengths[clip]-1)
            self.motion_fraction[ids]=self.reference_blend[clip,frame]

    def observe(self):
        base=super().observe();idx=torch.minimum(self.frames,self.lengths[self.clips]-1)
        return torch.cat((base,self.body_inputs[self.clips,idx],self.motion_fraction[:,None]),-1)

    @torch.no_grad()
    def step(self,targets):
        idx=torch.minimum(self.frames,self.lengths[self.clips]-1);clip=self.clips
        joint=self.references['joint'][clip,idx]
        near=torch.minimum((joint-self.references['joint'][clip,0]).abs().amax(-1),(joint-self.default).abs().amax(-1))<=.04
        moving=(~near)|(self.references['joint_velocity'][clip,idx].abs().amax(-1)>.10)|(self.references['root_velocity'][clip,idx].norm(dim=-1)>.025)|(self.references['root_omega'][clip,idx].norm(dim=-1)>.10)
        desired=moving&(self.frames<self.fault_frames[clip])
        self.motion_fraction+=(desired.float()-self.motion_fraction).clamp(-.04,.04)
        return super().step(targets)
