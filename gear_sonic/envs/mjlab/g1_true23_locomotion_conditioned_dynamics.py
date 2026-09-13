"""Received-body dynamics around native factory locomotion; no future input."""
from pathlib import Path
import numpy as np
import torch
from gear_sonic.envs.mjlab.g1_true23_factory_dynamics import FactoryNative23Env
from gear_sonic.envs.mjlab.g1_true23_causal_dynamics import features_torch,quaternion_matrix


class LocomotionConditionedEnv(FactoryNative23Env):
    def __init__(self,bundle,bank,config,count=128,device='cuda:0',command_delay_substeps=0,canonical_starts=False,canonical_worlds=2,physics_backend='warp'):
        super().__init__(bundle,bank,config,count,device,physics_backend=physics_backend)
        self.action_scale=self.old_scale.clone()
        self.loco_history=torch.zeros((count,4,42),device=device)
        self.loco_previous=torch.zeros((count,12),device=device)
        self.command_velocity=torch.zeros((count,3),device=device)
        self.loco_phase=torch.zeros(count,device=device)
        self.loco_walking=torch.zeros(count,device=device,dtype=torch.bool)
        self.command_delay_substeps=int(command_delay_substeps)
        self.canonical_starts=bool(canonical_starts)
        self.canonical_roles=int(canonical_worlds)
        if not 1<=self.canonical_roles<=6:raise ValueError('canonical worlds must occupy1..6 of each8 roles')
        self.canonical_world=torch.zeros(count,device=device,dtype=torch.bool)
        training_clips=torch.tensor(list(self.pools),device=device)
        self.canonical_assignment=training_clips[torch.arange(count,device=device)%len(training_clips)]
        self.reset_teacher_valid=torch.ones(count,device=device,dtype=torch.bool)
        if not 0<=self.command_delay_substeps<=3:raise ValueError('command delay must be0..3 physics substeps')
        self.last_applied=self.q.new_zeros((count,23));self.held_for_delay=self.last_applied.clone()
        self.delay_steps=torch.zeros(count,device=device,dtype=torch.long);self.control_substep=0
        with np.load(Path(bank)/'bfm_reference_inputs_v1.npz',allow_pickle=False) as z:
            if not bool(z['task_closure_stand']):raise ValueError('requires received geometric closure bank')
            length=self.states.shape[1]
            alpha=[np.pad(z[f'alpha_{i}'],(0,length-len(z[f'alpha_{i}'])),mode='edge') for i in range(len(self.meta['clips']))]
        self.loco_alpha=torch.tensor(np.stack(alpha),device=device)
        self.loco_ready=True;self.source_duration=200
        self.reset(torch.arange(count,device=device))

    def native_prop(self):
        return torch.cat((self.v[:,3:6]*.2,-quaternion_matrix(self.q[:,3:7])[:,2],
            self.q[:,7:19]-self.factory_default[:12],self.v[:,6:18]*.05,self.loco_previous),1).clamp(-10,10)

    def control_fields(self):
        f=features_torch(self.q,self.v,self.references,self.frames,self.clips,self.default,
            self.prior,self.history_vector(),self.lengths)
        xy=f[:,111:113]+2.5*f[:,102:104]
        yaw=torch.atan2(f[:,107],f[:,105])
        wanted=torch.cat((xy,f[:,116:117]+3*yaw[:,None]),1)
        cap=f.new_tensor([1.2,1.,6.]);rate=f.new_tensor([.24,.24,.8])
        wanted=wanted.clamp(-cap,cap)
        velocity=self.command_velocity+(wanted-self.command_velocity).clamp(-rate,rate)
        magnitude=torch.maximum(velocity[:,:2].norm(dim=-1),velocity[:,2].abs())
        walking=torch.where(self.loco_walking,magnitude>=.06,magnitude>.12)
        frequency=walking.float()*1.2
        phase=torch.where(walking,(self.loco_phase+frequency*.02)%1,0.)
        feet_phase=(phase[:,None]+f.new_tensor([0.,.5]))%1
        feet_phase=torch.where(walking[:,None],feet_phase,0.)
        command=torch.cat((torch.sin(2*torch.pi*feet_phase),torch.cos(2*torch.pi*feet_phase),
            (frequency[:,None]-1.2)*.5,velocity*.2),1)
        prop=self.native_prop()
        native=torch.cat((self.loco_history,prop[:,None]),1).flatten(1)
        index=torch.minimum(self.frames,self.lengths[self.clips]-1)
        x=torch.cat((native,command,f,self.loco_alpha[self.clips,index,None]),1)
        return x,prop,velocity,phase,walking

    def observe(self):
        if not getattr(self,'loco_ready',False):return super().observe()
        return self.control_fields()[0]

    def reset(self,ids,canonical=False):
        super().reset(ids,canonical)
        if not getattr(self,'loco_ready',False) or not len(ids):return
        self.reset_teacher_valid[ids]=True
        if self.canonical_starts:
            group=ids[(self.roles[ids]>=6-self.canonical_roles)&(self.roles[ids]<6)]
            if len(group):
                self.sim.reset(group)
                # Fixed per-world recording quotas keep a long, easier Pico
                # episode from occupying worlds needed for the shorter walks.
                self.clips[group]=self.canonical_assignment[group]
                state=self.states[self.clips[group],10]
                self.q[group]=state[:,:30];self.v[group]=state[:,30:]
                self.v[group,:2]+=.06*(torch.rand((len(group),2),device=self.device)-.5)
                self.frames[group]=11;self.prior[group]=0;self.factory_prior[group]=0
                for history in self.history:history[group]=0
                self.episode_end_frames[group]=10**9
                self.episode_control_limits[group]=self.totals[self.clips[group]]+1500
                self.canonical_world[group]=True
                # The parent restored an expert row before we replaced its
                # state with the actual start. Its target belongs to that old
                # state and must never supervise this new observation.
                self.reset_teacher_valid[group]=False
                self.past_proprio[group]=self.proprioception()[group,None]
                self.sim.forward()
        self.loco_previous[ids]=self.factory_prior[ids,:12]*.25
        self.command_velocity[ids]=0
        self.loco_phase[ids]=torch.rand(len(ids),device=self.device)
        self.loco_phase[ids]=torch.where(self.canonical_world[ids],0.,self.loco_phase[ids])
        self.loco_walking[ids]=False
        self.loco_history[ids]=self.native_prop()[ids,None]
        self.last_applied[ids]=self.factory_default+self.factory_prior[ids][:,self.q.new_tensor(list(range(13))+list(range(15,20))+list(range(22,27)),dtype=torch.long)]*.25

    def control_torque(self,targets):
        if not getattr(self,'loco_ready',False) or not self.command_delay_substeps:return super().control_torque(targets)
        delayed=self.control_substep<self.delay_steps
        applied=torch.where(delayed[:,None],self.held_for_delay,targets)
        self.control_substep+=1
        return super().control_torque(applied)

    def step(self,targets):
        self.held_for_delay.copy_(self.last_applied);self.last_applied.copy_(targets)
        self.control_substep=0
        if self.command_delay_substeps:self.delay_steps.random_(1,self.command_delay_substeps+1)
        # The independent plant's first target is warmed before its clock starts.
        self.delay_steps[self.canonical_world&(self.age==0)]=0
        _,prop,velocity,phase,walking=self.control_fields()
        self.loco_history[:,:-1]=self.loco_history[:,1:].clone();self.loco_history[:,-1]=prop
        self.command_velocity[:]=velocity;self.loco_phase[:]=phase;self.loco_walking[:]=walking
        self.loco_previous[:]=targets[:,:12]-self.factory_default[:12]
        return super().step(targets)
