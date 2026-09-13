"""Full-range native23 lifecycle environment on the existing MJLab simulator.

This is a distinct dynamics-learning experiment, not the bounded BFM residual
recipe. Physics uses the prepared native model, original PD/effort limits and
ten 2ms steps. Reference resets are training-only and separately counted.
"""
import json
from pathlib import Path
import numpy as np
import mujoco
import torch
from torch import nn
from gear_sonic.utils.g1_true23_received_features import HISTORY_WIDTHS


def quaternion_matrix(q):
    w,x,y,z=q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
        2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
        2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1],3,3)


def received_reference_stationary(reference):
    """Standing rewards require every received body objective to be stationary."""
    r=reference
    return ((r['root_velocity'].norm(dim=-1)<.02)
        &(r['joint_velocity'].abs().amax(-1)<.05)
        &(r['root_omega'].norm(dim=-1)<.05)
        &(r['feet_velocity'].norm(dim=-1).amax(-1)<.02)
        &(r['task_velocity'].norm(dim=-1).amax(-1)<.02)
        &(r['task_omega'].norm(dim=-1).amax(-1)<.05))


def features_torch(q, v, references, frames, clips, default, prior, history, lengths):
    offsets=frames.new_tensor([0,-1,-2,-4,-8,-16,-24,-37])
    slots=(frames[:,None]+offsets).clamp_min(0)
    slots=torch.minimum(slots,lengths[clips,None]-1)
    r={k:x[clips[:,None],slots] for k,x in references.items()}
    rot=quaternion_matrix(q[:,3:7]); yaw=torch.atan2(rot[:,1,0],rot[:,0,0])
    c,s=torch.cos(yaw),torch.sin(yaw); z=torch.zeros_like(c); o=torch.ones_like(c)
    h=torch.stack((c,-s,z,s,c,z,z,z,o),-1).reshape(-1,3,3)
    def vec(x): return torch.einsum('n...i,nij->n...j',x,h)
    def mat(x): return torch.einsum('nji,n...jk->n...ik',h,x)[...,:2]
    root=q[:,:3]
    proprio=torch.cat((q[:,7:]-default,v[:,6:],v[:,3:6],-rot[:,2,:],vec(v[:,:3]),q[:,2:3]),-1)
    goal=torch.cat((r['joint'],r['joint_velocity'],vec(r['root']-root[:,None]),mat(r['root_rotation']).flatten(-2),
        vec(r['root_velocity']),vec(r['root_omega']),vec(r['feet']-root[:,None,None]).flatten(-2),
        vec(r['feet_velocity']).flatten(-2),vec(r['tasks']-root[:,None,None]).flatten(-2),
        mat(r['task_rotation']).flatten(-3),vec(r['task_velocity']).flatten(-2),vec(r['task_omega']).flatten(-2)),-1)
    return torch.cat((proprio,goal.flatten(1),prior,history),-1)


class Actor(nn.Module):
    def __init__(self, checkpoint, projected_mean=False):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(1323,512),nn.ELU(),nn.Linear(512,512),nn.ELU(),nn.Linear(512,23))
        self.net.load_state_dict(checkpoint['actor_state'])
        self.register_buffer('mean',checkpoint['feature_mean'].float().clone())
        self.register_buffer('scale',checkpoint['feature_std'].float().clone())
        self.register_buffer('span',checkpoint['joint_span'].float().clone())
        self.register_buffer('default',checkpoint['default_q'].float().clone())
        self.register_buffer('limits',checkpoint['joint_limits'].float().clone())
        self.projected_mean=projected_mean

    def forward(self,x):
        raw=self.net((x-self.mean)/self.scale)
        if not self.projected_mean:return raw
        lower=(self.limits[:,0]-self.default)/self.span
        upper=(self.limits[:,1]-self.default)/self.span
        projected=raw.clamp(lower,upper)
        # Forward distribution is centered on a legal command, not far beyond
        # the actuator clamp. Straight-through gradients are an explicitly
        # experimental surrogate for this projection; native rollouts decide.
        return raw+(projected-raw).detach()

    def target(self,action):
        return torch.clamp(self.default+self.span*action,self.limits[:,0],self.limits[:,1])


class CausalNative23Env:
    def __init__(self, bundle, bank, count=128, device='cuda:0', seed=20260912,physics_backend='warp'):
        torch.manual_seed(seed)
        self.device,self.count,self.bank= device,count,Path(bank)
        self.meta=json.loads((self.bank/'bank.json').read_text())
        if (self.bank/'interruptions.json').exists():
            self.meta['clips']+=json.loads((self.bank/'interruptions.json').read_text())['clips']
        self.c=json.loads((Path(bundle)/'contract.json').read_text())
        model=mujoco.MjModel.from_xml_path(str(Path(bundle)/'native_prepared.xml'))
        with np.load(Path(bundle)/'prepared_model_arrays.npz') as z:
            for k in z.files: getattr(model,k)[:]=z[k]
        mujoco.mj_setConst(model,mujoco.MjData(model))
        # Simulation normally applies defaults. Preserve every existing opt.
        opt=model.opt
        if physics_backend=='mjbatch':
            from gear_sonic.envs.mjlab.g1_true23_mjbatch_sim import NativeBatchSimulation
            self.sim=NativeBatchSimulation(count,model,device,self.c['native_velocity'])
        elif physics_backend=='warp':
            from mjlab.sim.sim import Simulation, SimulationCfg, MujocoCfg
            simcfg=SimulationCfg(nconmax=128,njmax=512,mujoco=MujocoCfg(
                timestep=float(opt.timestep), integrator='euler', impratio=float(opt.impratio),
                cone='elliptic' if opt.cone else 'pyramidal', jacobian='auto',
                solver={0:'pgs',1:'cg',2:'newton'}[int(opt.solver)],iterations=int(opt.iterations),
                tolerance=float(opt.tolerance),ls_iterations=int(opt.ls_iterations),ls_tolerance=float(opt.ls_tolerance),
                ccd_iterations=int(opt.ccd_iterations),gravity=tuple(opt.gravity)))
            self.sim=Simulation(count,simcfg,model,device)
        else:raise ValueError('unknown physics backend')
        self.q,self.v=self.sim.data.qpos,self.sim.data.qvel
        self.default=torch.tensor(self.c['default_q'],device=device)
        self.kp=torch.tensor(self.c['kp'],device=device);self.kd=torch.tensor(self.c['kd'],device=device)
        self.effort=torch.tensor(self.c['native_effort'],device=device)
        self.velocity=torch.tensor(self.c['native_velocity'],device=device)
        self.limits=torch.tensor(self.c['joint_limits'],device=device)
        self.action_scale=.25*torch.tensor(self.c['training_effort'],device=device)/self.kp
        self.feet=[model.body(s+'_ankle_roll_link').id for s in ('left','right')]
        self.task_ids=[model.body(t['target_body']).id for t in self.meta['tasks']]
        self.task_offsets=torch.tensor([t['target_point'] for t in self.meta['tasks']],device=device)
        parts=[]
        for row in self.meta['clips']:
            with np.load(self.bank/row['file']) as z: parts.append({k:z[k].copy() for k in z.files})
        self.lengths=torch.tensor([len(x['joint']) for x in parts],device=device)
        n=int(self.lengths.max())
        self.references={k:torch.tensor(np.stack([np.concatenate((x[k],np.repeat(x[k][-1:],n-len(x[k]),axis=0))) for x in parts]),
                                       device=device,dtype=torch.float32) for k in parts[0] if k!='states'}
        self.states=torch.tensor(np.stack([np.concatenate((x['states'],np.repeat(x['states'][-1:],n-len(x['states']),axis=0))) for x in parts]),
                                  device=device,dtype=torch.float32)
        self.totals=torch.tensor([r['total'] for r in self.meta['clips']],device=device)
        self.source_starts=torch.tensor([r['source_start']+11 for r in self.meta['clips']],device=device)
        self.source_stops=torch.tensor([r['source_stop']+11 for r in self.meta['clips']],device=device)
        self.training_ids=torch.tensor([i for i,r in enumerate(self.meta['clips']) if r['training']],device=device)
        self.frames=torch.zeros(count,dtype=torch.long,device=device)
        self.clips=torch.zeros_like(self.frames)
        self.age=torch.zeros_like(self.frames)
        self.prior=torch.zeros((count,23),device=device)
        self.history=[torch.zeros((count,4,w),device=device) for w in HISTORY_WIDTHS]
        self.bad=torch.zeros(count,dtype=torch.bool,device=device)
        self.episode_return=torch.zeros(count,device=device)
        with np.load(self.bank/'expert.npz') as z:
            self.expert_features=torch.tensor(z['features'],device=device)
            self.expert_targets=torch.tensor(z['targets'],device=device)
            self.expert_resets=torch.tensor(z['resets'],device=device,dtype=torch.float32)
        self.reset_count=0
        self.reset(torch.arange(count,device=device),canonical=True)

    def history_vector(self): return torch.cat([x.flatten(1) for x in self.history],-1)

    def motion_prior_observation(self):
        from gear_sonic.utils.g1_true23_motion_prior import physical_features
        mats=self.sim.data.xmat[:,self.task_ids].reshape(self.count,3,3,3)
        tasks=self.sim.data.xpos[:,self.task_ids]+torch.einsum('ntij,tj->nti',mats,self.task_offsets)
        return physical_features(self.q,self.v,self.sim.data.xpos[:,self.feet],tasks)

    def observe(self):
        return features_torch(self.q,self.v,self.references,self.frames,self.clips,self.default,
                              self.prior,self.history_vector(),self.lengths)

    def reset(self,ids,canonical=False):
        if not len(ids): return
        self.sim.reset(ids)
        n=len(ids)
        clips=self.training_ids[torch.randint(len(self.training_ids),(n,),device=self.device)]
        self.clips[ids]=clips
        self.frames[ids]=11
        self.q[ids]=self.states[clips,10,:30]
        self.v[ids]=self.states[clips,10,30:]
        self.prior[ids]=0
        for h in self.history: h[ids]=0
        if not canonical:
            # Half true lifecycle starts; half actual expert states, including
            # terminal standing. Never use held-out walk008 training states.
            selected=ids[torch.rand(n,device=self.device)<.5]
            ix=torch.randint(len(self.expert_resets),(len(selected),),device=self.device)
            rows=self.expert_resets[ix]
            self.clips[selected]=rows[:,0].long();self.frames[selected]=rows[:,1].long()
            self.q[selected]=rows[:,2:32];self.v[selected]=rows[:,32:61]
            self.prior[selected]=rows[:,61:84]
            offset=84
            for h,width in zip(self.history,HISTORY_WIDTHS):
                h[selected]=rows[:,offset:offset+4*width].reshape(-1,4,width);offset+=4*width
            self.v[ids,:2]+=.06*(torch.rand((n,2),device=self.device)-.5)
        self.age[ids]=0;self.bad[ids]=False;self.episode_return[ids]=0
        self.sim.forward()
        self.reset_count+=n

    def control_torque(self, targets):
        return torch.clamp(self.kp*(targets-self.q[:,7:])-self.kd*self.v[:,6:],-self.effort,self.effort)

    @torch.no_grad()
    def step(self, targets):
        q0,v0=self.q.clone(),self.v.clone()
        gravity=-quaternion_matrix(q0[:,3:7])[:,2,:]
        terms=(self.prior,v0[:,3:6]*.25,q0[:,7:]-self.default,v0[:,6:],gravity)
        for h,t in zip(self.history,terms): h[:,1:]=h[:,:-1].clone();h[:,0]=t
        self.prior.copy_((targets-self.default)/self.action_scale)
        self.bad.zero_()
        speed_peak=torch.zeros(self.count,device=self.device)
        for _ in range(10):
            self.sim.data.ctrl.copy_(self.control_torque(targets))
            self.sim.step()
            if hasattr(self.sim.data,'physical_bad'):self.bad|=self.sim.data.physical_bad
            speed=(self.v[:,6:].abs()/self.velocity).amax(-1)
            speed_peak=torch.maximum(speed_peak,speed)
            self.bad|=((self.q[:,7:]<self.limits[:,0]-1e-6)|(self.q[:,7:]>self.limits[:,1]+1e-6)).any(-1)
            self.bad|=(speed>1)|(~torch.isfinite(self.q).all(-1))|(~torch.isfinite(self.v).all(-1))
        # Refresh kinematics for the actual post-integration state, then score
        # before any reset. Native CPU evaluation remains final authority.
        self.sim.forward()
        rot=quaternion_matrix(self.q[:,3:7])
        tilt=torch.acos(rot[:,2,2].clamp(-1,1))
        self.bad|=(self.q[:,2]<.25)|(tilt>1.2)
        idx=torch.minimum(self.frames,self.lengths[self.clips]-1)
        r={k:x[self.clips,idx] for k,x in self.references.items()}
        root=self.q[:,:3]
        feet=self.sim.data.xpos[:,self.feet]
        mats=self.sim.data.xmat[:,self.task_ids].reshape(self.count,3,3,3)
        tasks=self.sim.data.xpos[:,self.task_ids]+torch.einsum('ntij,tj->nti',mats,self.task_offsets)
        root_error=(root-r['root']).square().sum(-1)
        dy=torch.atan2(rot[:,1,0],rot[:,0,0])-torch.atan2(r['root_rotation'][:,1,0],r['root_rotation'][:,0,0])
        yaw=torch.atan2(torch.sin(dy),torch.cos(dy))
        foot_error=((feet-root[:,None])-(r['feet']-r['root'][:,None])).square().sum(-1)
        task_error=((tasks-root[:,None])-(r['tasks']-r['root'][:,None])).square().sum(-1)
        leg_error=(self.q[:,7:19]-r['joint'][:,:12]).square().mean(-1)
        joint_error=(self.q[:,7:]-r['joint']).square().mean(-1)
        quiet=received_reference_stationary(r)
        reward=.1*(1+2*torch.exp(-root_error/.04)+torch.exp(-yaw.square()/.0685)
            +2*torch.exp(-foot_error/.0144).mean(-1)+2*torch.exp(-task_error[:,:2]/.0225).mean(-1)
            +torch.exp(-task_error[:,2]/.01)+3*torch.exp(-leg_error/.0225)+torch.exp(-joint_error/.04)
            +quiet*(torch.exp(-self.v[:,:3].square().sum(-1)/.0025)+torch.exp(-self.v[:,6:].square().mean(-1)/.25)))
        reward-=.01*((targets-q0[:,7:])/self.action_scale).square().mean(-1).clamp_max(100)
        reward-=.1*torch.relu(speed_peak-.8).square()
        source_motion=(self.frames>=self.source_starts[self.clips])&(self.frames<self.source_stops[self.clips])
        tracking_failed=torch.zeros_like(self.bad)
        if getattr(self,'focused_tracking',False):
            # Training-only signal/termination, distinct from native physical
            # failures and the unchanged complete-motion acceptance limits.
            # Exponential rewards otherwise become nearly flat far off track.
            cost=(root_error/.04+foot_error.mean(-1)/.0144+
                  task_error[:,:2].mean(-1)/.0225+task_error[:,2]/.01+leg_error/.0225)/5
            reward-=source_motion.float()*.15*cost.clamp_max(20)
            tracking_failed=source_motion&((root_error>.5**2)|(leg_error>.45**2)|
                (foot_error.amax(-1)>.35**2)|(task_error.amax(-1)>.55**2))
            if getattr(self,'terminate_tracking_errors',True):
                reward=torch.where(tracking_failed,torch.full_like(reward,-5),reward)
        settling=getattr(self,'settle_objective',False)
        if settling:
            # Received velocities identify a held pose, without a clip phase
            # or final-frame flag. Keep every existing body tracking reward.
            # Dense costs remain informative after exponential rewards flatten.
            yaw_cost=yaw.square()/.0685
            stopped_cost=(self.v[:,:3].square().sum(-1)/.0025+
                          self.v[:,6:].square().mean(-1)/.25+tilt.square()/.0225)/3
            pose_cost=(root_error/.04+foot_error.mean(-1)/.0144+
                       task_error[:,:2].mean(-1)/.0225+task_error[:,2]/.01+leg_error/.0225)/5
            reward-=.15*yaw_cost.clamp_max(20)
            reward-=quiet.float()*(.3*stopped_cost.clamp_max(20)+.15*pose_cost.clamp_max(20))
        continue_tracking=not getattr(self,'terminate_tracking_errors',True)
        if continue_tracking:
            # Preserve every dense task cost, while keeping surviving-step
            # reward positive. This avoids teaching the agent to end an episode
            # by falling instead of recovering from a prolonged tracking error.
            reward=torch.nn.functional.softplus(reward)
        reward=torch.where(self.bad,torch.full_like(reward,-10),reward)
        self.age+=1;self.frames+=1;self.episode_return+=reward
        complete=self.frames>=self.totals[self.clips]+11+getattr(self,'training_extra_hold_controls',1500)
        truncated=torch.zeros_like(complete)
        tracking_terminal=tracking_failed if not continue_tracking else torch.zeros_like(tracking_failed)
        if getattr(self,'quota_ready',False):
            # Source/transition assignments have time limits, not successful
            # lifecycle terminals. Capture their actual next observation before
            # resetting so PPO can bootstrap without crossing into a new episode.
            cutoff=(self.frames>=self.episode_end_frames)|(self.age>=self.episode_control_limits)
            truncated=cutoff&~(self.bad|tracking_terminal|complete)
        done=self.bad|tracking_terminal|complete|truncated
        settled=(quiet&(self.v[:,:3].norm(dim=-1)<=.05)&
                 (self.v[:,6:].abs().amax(-1)<=2)&(tilt<=.15)&
                 (root_error<=.05**2)&(yaw.abs()<=.0872665))
        # Reaching the episode clock while drifting is not successful standing.
        # This is a training reward only; native evaluation still requires the
        # entire uninterrupted 30-second hold and its existing quiet window.
        rewarded_complete=complete&settled&~self.bad if settling else complete
        reward+=rewarded_complete.float()*5
        info=dict(failed=self.bad.clone(),complete=complete.clone(),age=self.age.clone(),
                  root_error=root_error.sqrt(),leg_error=leg_error.sqrt(),foot_error=foot_error.sqrt().amax(-1),
                  hand_error=task_error[:,:2].sqrt().amax(-1),head_error=task_error[:,2].sqrt(),
                  source_motion=source_motion,tracking_failed=tracking_failed,
                  settled=settled,rewarded_complete=rewarded_complete,
                  terminated=self.bad|tracking_terminal|complete,truncated=truncated,
                  yaw_error=yaw.abs(),foot_errors=foot_error.sqrt(),task_errors=task_error.sqrt(),
                  speed_ratio=speed_peak,episode_return=self.episode_return.clone())
        if truncated.any():info['final_observation']=self.observe().clone()
        if getattr(self,'capture_evaluation_state',False):
            info['physical_qpos']=self.sim.native['qpos'].copy()
            info['physical_qvel']=self.sim.native['qvel'].copy()
            info['reference_root']=r['root'].cpu().numpy().copy()
        if getattr(self,'capture_motion_prior',False):
            # Capture the physical successor, including terminal states, before
            # reset replaces it. AMP must never learn reset jumps as dynamics.
            info['motion_prior_next']=self.motion_prior_observation().clone()
        self.reset(done.nonzero(as_tuple=False).flatten())
        return self.observe(),reward,done,info
