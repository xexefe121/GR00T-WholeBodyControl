"""Received-only full-range takeover training with retained neutral balance."""
from pathlib import Path
import numpy as np
import torch
from gear_sonic.envs.mjlab.g1_true23_causal_dynamics import CausalNative23Env
from gear_sonic.utils.g1_true23_neutral_balance import NeutralBalance
from gear_sonic.utils.g1_true23_bfm_residual import heading_matrix
from gear_sonic.utils.g1_true23_received_features import HISTORY_WIDTHS


class BalancedCausalNative23Env(CausalNative23Env):
    def __init__(self,bundle,bank,count=128,device='cuda:0',seed=20260912):
        super().__init__(bundle,bank,count,device,seed)
        with np.load(Path(bundle)/'walk003/native_original.npz') as z:
            neutral={k:z[k][:1].copy() for k in z.files if k!='fps'}
        root=Path(__file__).resolve().parents[3]
        weights=root/'artifacts/g1_true23_six_hour_replan_20260910_v1/bfmzero_inference_v1/inference.safetensors'
        self.balance=NeutralBalance(weights,self.c,neutral,device)
        self.neutral=torch.tensor(neutral['joint_pos'][0],dtype=torch.float32,device=device)
        self.motion_fraction=torch.zeros(count,device=device)
        self.fault=torch.zeros(count,dtype=torch.bool,device=device)
        self.anchor=torch.zeros(count,3,device=device)
        self.fault_frames=torch.tensor([r['source_stop']+11 if r.get('generated_interruption') else 10**9 for r in self.meta['clips']],device=device)
        self.recovery_pools=[]
        for i,r in enumerate(self.meta['clips']):
            if not r.get('generated_interruption'):continue
            original=next(j for j,c in enumerate(self.meta['clips'][:3]) if r['name'].startswith(c['name']+'_'))
            cut=r['source_stop']+11
            pool=((self.expert_resets[:,0]==original)&(self.expert_resets[:,1]>=cut-20)&(self.expert_resets[:,1]<cut)).nonzero().flatten()
            if not len(pool):raise ValueError('missing actual recovery states for '+r['name'])
            self.recovery_pools.append((i,pool))
        self.reset(torch.arange(count,device=device),canonical=True)

    def restore_rows(self,ids,rows,clip=None):
        self.clips[ids]=rows[:,0].long() if clip is None else clip
        self.frames[ids]=rows[:,1].long();self.q[ids]=rows[:,2:32];self.v[ids]=rows[:,32:61]
        self.prior[ids]=rows[:,61:84];offset=84
        for h,width in zip(self.history,HISTORY_WIDTHS):
            h[ids]=rows[:,offset:offset+4*width].reshape(-1,4,width);offset+=4*width

    def reset(self,ids,canonical=False):
        if not hasattr(self,'recovery_pools'):return super().reset(ids,canonical)
        if not len(ids):return
        self.sim.reset(ids);n=len(ids)
        self.clips[ids]=self.training_ids[torch.randint(len(self.training_ids),(n,),device=self.device)]
        self.frames[ids]=11
        self.q[ids]=self.states[self.clips[ids],10,:30];self.v[ids]=self.states[self.clips[ids],10,30:]
        self.prior[ids]=0
        for h in self.history:h[ids]=0
        if not canonical:
            roles=torch.randint(3,(n,),device=self.device)
            expert_ids=ids[roles==1]
            rows=self.expert_resets[torch.randint(len(self.expert_resets),(len(expert_ids),),device=self.device)]
            self.restore_rows(expert_ids,rows)
            recovery_ids=ids[roles==2]
            variants=torch.randint(len(self.recovery_pools),(len(recovery_ids),),device=self.device)
            for j,(clip,pool) in enumerate(self.recovery_pools):
                chosen=recovery_ids[variants==j]
                rows=self.expert_resets[pool[torch.randint(len(pool),(len(chosen),),device=self.device)]]
                self.restore_rows(chosen,rows,clip)
            self.v[ids,:2]+=.06*(torch.rand(n,2,device=self.device)-.5)
        frame=self.frames[ids];clip=self.clips[ids]
        reference=self.references['joint'][clip,frame]
        # Actual moving expert starts begin in full motion control. Canonical
        # standing starts retain BFM until received pose begins changing.
        moving=(reference-self.references['joint'][clip,0]).abs().amax(-1)>.04
        self.motion_fraction[ids]=moving.float() if not canonical else 0
        self.fault[ids]=False;self.anchor[ids]=0
        self.age[ids]=0;self.bad[ids]=False;self.episode_return[ids]=0
        self.sim.forward();self.reset_count+=n

    @torch.no_grad()
    def step(self,targets):
        idx=torch.minimum(self.frames,self.lengths[self.clips]-1)
        joint=self.references['joint'][self.clips,idx]
        first=self.references['joint'][self.clips,0]
        speed=self.references['joint_velocity'][self.clips,idx].abs().amax(-1)
        standing=torch.minimum((joint-first).abs().amax(-1),(joint-self.neutral).abs().amax(-1))<=.04
        root_speed=self.references['root_velocity'][self.clips,idx].norm(dim=-1)
        root_turn=self.references['root_omega'][self.clips,idx].norm(dim=-1)
        desired=(~standing)|(speed>.10)|(root_speed>.025)|(root_turn>.10)
        fault_now=self.frames>=self.fault_frames[self.clips]
        first_fault=fault_now&~self.fault
        _,actual_yaw=heading_matrix(self.q[:,3:7])
        self.anchor[first_fault,:2]=self.q[first_fault,:2]
        self.anchor[first_fault,2]=actual_yaw[first_fault]
        self.fault|=fault_now;desired&=~self.fault
        self.motion_fraction+=(desired.float()-self.motion_fraction).clamp(-.04,.04)
        fraction=self.motion_fraction.clone()
        rotation=self.references['root_rotation'][self.clips,idx]
        reference_anchor=torch.cat((self.references['root'][self.clips,idx,:2],torch.atan2(rotation[:,1,0],rotation[:,0,0])[:,None]),-1)
        anchor=torch.where(self.fault[:,None],self.anchor,reference_anchor)
        balance=self.balance.target(self.q,self.v,self.prior,self.history_vector(),anchor)
        applied=(1-fraction[:,None])*balance+fraction[:,None]*targets
        obs,reward,done,info=super().step(applied)
        info['motion_fraction']=fraction
        return obs,reward,done,info
