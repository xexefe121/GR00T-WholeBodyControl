"""Full current-body goal encoding with the same native dynamics and resets."""
import numpy as np
import torch
from gear_sonic.envs.mjlab.g1_true23_single_policy_dynamics import SinglePolicyNative23Env


class DirectBodyNative23Env(SinglePolicyNative23Env):
    def __init__(self,*args,task_closure_stand=False,**kwargs):
        self.task_closure_stand=bool(task_closure_stand)
        super().__init__(*args,**kwargs)
        if bool(self.meta.get('task_closure_stand',False))!=self.task_closure_stand:
            raise ValueError('reference bank task-closure semantics differ from requested environment')
        with np.load(self.bank/'bfm_reference_inputs_v1.npz') as z:
            if bool(z['task_closure_stand'].item()) if 'task_closure_stand' in z else False:
                if not self.task_closure_stand:raise ValueError('task-closure body inputs require task-closure environment')
            elif self.task_closure_stand:raise ValueError('task-closure environment requires matching body inputs')
            maximum=int(self.lengths.max());rows=[];alphas=[];desires=[]
            for i in range(len(self.meta['clips'])):
                body=z[f'body_{i}'];alpha=z[f'alpha_{i}']
                rows.append(np.concatenate((body,np.repeat(body[-1:],maximum-len(body),0))))
                alphas.append(np.pad(alpha,(0,maximum-len(alpha)),mode='edge'))
                if self.task_closure_stand:desires.append(np.pad(z[f'desired_{i}'],(0,maximum-len(alpha)),mode='edge'))
        self.body_inputs=torch.tensor(np.stack(rows),device=self.device)
        self.reference_blend=torch.tensor(np.stack(alphas),device=self.device)
        if self.task_closure_stand:self.reference_desired=torch.tensor(np.stack(desires),device=self.device)
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
        if self.task_closure_stand:
            next_idx=torch.minimum(self.frames+1,self.lengths[clip]-1)
            following=self.reference_blend[clip,next_idx]
            # Beyond the last received sample, continue the same recurrence.
            # Interruption desire is precomputed as zero and stays latched.
            hold=self.motion_fraction+(self.reference_desired[clip,idx]-self.motion_fraction).clamp(-.04,.04)
            self.motion_fraction.copy_(torch.where(self.frames+1<self.lengths[clip],following,hold))
            return super().step(targets)
        joint=self.references['joint'][clip,idx]
        near=torch.minimum((joint-self.references['joint'][clip,0]).abs().amax(-1),(joint-self.default).abs().amax(-1))<=.04
        moving=(~near)|(self.references['joint_velocity'][clip,idx].abs().amax(-1)>.10)|(self.references['root_velocity'][clip,idx].norm(dim=-1)>.025)|(self.references['root_omega'][clip,idx].norm(dim=-1)>.10)
        desired=moving&(self.frames<self.fault_frames[clip])
        self.motion_fraction+=(desired.float()-self.motion_fraction).clamp(-.04,.04)
        return super().step(targets)
