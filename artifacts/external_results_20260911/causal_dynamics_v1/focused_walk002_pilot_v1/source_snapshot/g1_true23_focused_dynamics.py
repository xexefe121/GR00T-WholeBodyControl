"""Motion-focused native dynamics pilot; evaluation remains the full lifecycle."""
import torch
from gear_sonic.envs.mjlab.g1_true23_direct_body_dynamics import DirectBodyNative23Env


class FocusedDirectBodyNative23Env(DirectBodyNative23Env):
    def __init__(self,bundle,bank,count=64,focus_clip='walk002',**kwargs):
        super().__init__(bundle,bank,count,**kwargs)
        names=[row['name'] for row in self.meta['clips']]
        if focus_clip not in names[:3]:raise ValueError('focus must be a training clip')
        self.focus_id=names.index(focus_clip)
        self.focus_training_ids=torch.tensor([i for i,row in enumerate(self.meta['clips'])
            if row['training'] and (row['name']==focus_clip or row['name'].startswith(focus_clip+'_'))],device=self.device)
        self.focus_recovery_pools=[(i,pool) for i,pool in self.recovery_pools
                                   if names[i].startswith(focus_clip+'_')]
        row=self.meta['clips'][self.focus_id]
        starts,stops=row['source_start']+11,row['source_stop']+11
        same=self.expert_resets[:,0]==self.focus_id
        self.focus_motion_pool=(same&(self.expert_resets[:,1]>=starts)&(self.expert_resets[:,1]<stops)).nonzero().flatten()
        self.focus_braking_pool=(same&(self.expert_resets[:,1]>=stops-50)).nonzero().flatten()
        if not len(self.focus_motion_pool) or not len(self.focus_braking_pool):
            raise ValueError('focused pilot requires actual moving and braking expert states')
        self.focused_tracking=True
        self.focus_ready=True
        self.reset(torch.arange(count,device=self.device))

    def reset(self,ids,canonical=False):
        if not getattr(self,'focus_ready',False):return super().reset(ids,canonical)
        if not len(ids):return
        self.sim.reset(ids);n=len(ids)
        # 25% lifecycle starts; 50% real moving expert states; 25% braking or
        # interruption recovery states. No held-out frames or actor phase input.
        self.clips[ids]=self.focus_id;self.frames[ids]=11
        self.q[ids]=self.states[self.focus_id,10,:30];self.v[ids]=self.states[self.focus_id,10,30:]
        self.prior[ids]=0
        for history in self.history:history[ids]=0
        if not canonical:
            roles=torch.randint(4,(n,),device=self.device)
            moving=ids[(roles==1)|(roles==2)]
            pool=self.focus_motion_pool
            self.restore_rows(moving,self.expert_resets[pool[torch.randint(len(pool),(len(moving),),device=self.device)]])
            recovery=ids[roles==3]
            variants=torch.randint(len(self.focus_recovery_pools)+1,(len(recovery),),device=self.device)
            for option in range(len(self.focus_recovery_pools)+1):
                selected=recovery[variants==option]
                clip,pool=(self.focus_id,self.focus_braking_pool) if option==0 else self.focus_recovery_pools[option-1]
                rows=self.expert_resets[pool[torch.randint(len(pool),(len(selected),),device=self.device)]]
                self.restore_rows(selected,rows,clip)
            self.v[ids,:2]+=.06*(torch.rand((n,2),device=self.device)-.5)
        clip=self.clips[ids];frame=torch.minimum(self.frames[ids],self.lengths[clip]-1)
        self.motion_fraction[ids]=self.reference_blend[clip,frame]
        self.fault[ids]=False;self.anchor[ids]=0
        self.age[ids]=0;self.bad[ids]=False;self.episode_return[ids]=0
        self.sim.forward();self.reset_count+=n
