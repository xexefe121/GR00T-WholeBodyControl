"""Factory-initialized full-range native23 dynamics; training resets only."""
from pathlib import Path
import numpy as np
import mujoco
import torch
import yaml
from gear_sonic.envs.mjlab.g1_true23_causal_dynamics import CausalNative23Env, quaternion_matrix
from gear_sonic.utils.g1_true23_factory_policy import IDS


class FactoryNative23Env(CausalNative23Env):
    def __init__(self,bundle,bank,config,count=128,device='cuda:0',physics_backend='warp'):
        super().__init__(bundle,bank,count,device,physics_backend=physics_backend)
        cfg = yaml.safe_load(Path(config).read_text(encoding='utf-8'))
        self.old_kp,self.old_kd,self.old_scale = self.kp.clone(),self.kd.clone(),self.action_scale.clone()
        if 'default_joint_q' in cfg:
            defaults,kp,kd=cfg['default_joint_q'],cfg['joint_kp'],cfg['joint_kd']
            if len(defaults)==29:defaults,kp,kd=(np.asarray(v)[IDS] for v in (defaults,kp,kd))
        else:defaults,kp,kd=(np.asarray(cfg[key])[IDS] for key in ('default_dof_pos','kp','kd'))
        self.factory_default = torch.tensor(defaults,device=device,dtype=torch.float32)
        self.kp = torch.tensor(kp,device=device,dtype=torch.float32)
        self.kd = torch.tensor(kd,device=device,dtype=torch.float32)
        self.action_scale = torch.full((23,),.25,device=device)
        self.factory_prior = torch.zeros((count,29),device=device)
        self.past_proprio = torch.zeros((count,4,93),device=device)
        self.factory_ready = True
        self.focused_tracking = True
        self.settle_objective = True
        self.training_extra_hold_controls = 1500
        self.roles = torch.arange(count,device=device)%8  # six source, one transition, one standing
        self.episode_end_frames = torch.full_like(self.frames,10**9)
        self.episode_control_limits = torch.full_like(self.frames,1500)
        self.source_duration = 100
        self.pools = {}
        er = self.expert_resets
        self.teacher_table = self.q.new_zeros((len(self.meta['clips']),self.states.shape[1],23))
        self.teacher_table[er[:,0].long(),er[:,1].long()] = self.expert_targets
        self.reset_teacher = self.q.new_zeros((count,23))
        for clip,row in enumerate(self.meta['clips']):
            if not row['training'] or row.get('generated_interruption'): continue
            start,stop = row['source_start']+11,row['source_stop']+11
            same = er[:,0]==clip
            self.pools[clip] = {
                'source':(same&(er[:,1]>=start)&(er[:,1]<stop-100)).nonzero().flatten(),
                'acquire':(same&(er[:,1]>=start-100)&(er[:,1]<start)).nonzero().flatten(),
                'brake':(same&(er[:,1]>=stop-100)&(er[:,1]<stop)).nonzero().flatten(),
                'stand':(same&(er[:,1]>=stop+100)&(er[:,32:61].abs().amax(-1)<.5)).nonzero().flatten(),
            }
            if not all(len(self.pools[clip][key]) for key in ('source','acquire','brake')):
                raise ValueError(f'empty factory movement reset pool: {row["name"]}')
        self.standing_clips=[clip for clip,pool in self.pools.items() if len(pool['stand'])]
        if not self.standing_clips:raise ValueError('actual quiet standing examples are required')
        self.initial_standing_pool=((er[:,1]<self.source_starts[er[:,0].long()]-100)&
            (er[:,32:61].abs().amax(-1)<.5)).nonzero().flatten()
        if not len(self.initial_standing_pool):raise ValueError('actual initial standing examples are required')
        self.quota_ready = True
        # Reconstruct target quaternions from the recorded current pose only.
        rotations = self.references['root_rotation'].cpu().numpy()
        quats = np.empty((*rotations.shape[:2],4),np.float32)
        for index in np.ndindex(rotations.shape[:2]):
            q = np.empty(4); mujoco.mju_mat2Quat(q,rotations[index].astype(np.float64).ravel())
            if q[0]<0:q=-q
            quats[index] = q[[1,2,3,0]]
        self.reference_quat = torch.tensor(quats,device=device)
        self.reset(torch.arange(count,device=device))

    def brake_torque(self,q,v):
        band = torch.minimum(torch.full((23,),.1,device=self.device),(self.limits[:,1]-self.limits[:,0])*.2)
        penetration = q-torch.clamp(q,self.limits[:,0]+band,self.limits[:,1]-band)
        return 100*penetration+2*torch.where(penetration*v>0,v,torch.zeros_like(v))

    def control_torque(self,targets):
        torque = self.kp*(targets-self.q[:,7:])-self.kd*self.v[:,6:]
        return (torque-self.brake_torque(self.q[:,7:],self.v[:,6:])).clamp(-self.effort,self.effort)

    def proprioception(self):
        q29,v29 = self.q.new_zeros((self.count,29)),self.q.new_zeros((self.count,29))
        q29[:,IDS],v29[:,IDS] = self.q[:,7:],self.v[:,6:]
        return torch.cat((self.v[:,3:6],-quaternion_matrix(self.q[:,3:7])[:,2],q29,v29,self.factory_prior),-1)

    def observe(self):
        if not getattr(self,'factory_ready',False): return super().observe()
        index = torch.minimum(self.frames,self.lengths[self.clips]-1)
        r = {k:v[self.clips,index] for k,v in self.references.items()}
        prop = self.proprioception()
        memory = torch.cat((self.past_proprio,prop[:,None]),1).flatten(1)
        quat = self.q[:,3:7]*torch.where(self.q[:,3:4]<0,-1.,1.)
        q29,v29 = self.q.new_zeros((self.count,29)),self.q.new_zeros((self.count,29))
        q29[:,IDS],v29[:,IDS] = r['joint'],r['joint_velocity']
        rotation = quaternion_matrix(self.q[:,3:7])
        def local(x): return torch.einsum('n...i,nij->n...j',x,rotation)
        vel = torch.einsum('ni,nij->nj',r['root_velocity'],r['root_rotation'])
        omega = torch.einsum('ni,nij->nj',r['root_omega'],r['root_rotation'])
        target = torch.cat((r['root'],self.reference_quat[self.clips,index],q29,vel,omega,v29),-1)
        root = self.q[:,:3]
        feet = self.sim.data.xpos[:,self.feet]
        mats = self.sim.data.xmat[:,self.task_ids].reshape(self.count,3,3,3)
        tasks = self.sim.data.xpos[:,self.task_ids]+torch.einsum('ntij,tj->nti',mats,self.task_offsets)
        errors = torch.cat((local(r['root']-root)/.2,local(self.v[:,:3])/.3,
            (local((r['feet']-r['root'][:,None])-(feet-root[:,None]))/.12).flatten(1),
            (local((r['tasks']-r['root'][:,None])-(tasks-root[:,None]))/root.new_tensor([.15,.15,.1])[None,:,None]).flatten(1),
            self.q[:,2:3]/.8),-1)
        return torch.cat((memory,prop,quat[:,[1,2,3,0]],target,errors),-1)

    def restore(self,ids,rows,clip_override=None):
        self.clips[ids] = rows[:,0].long() if clip_override is None else clip_override
        self.frames[ids] = rows[:,1].long()
        self.q[ids],self.v[ids] = rows[:,2:32],rows[:,32:61]
        old_target = self.default+rows[:,61:84]*self.old_scale
        target = self.q[ids,7:]+(self.old_kp*(old_target-self.q[ids,7:])+(self.kd-self.old_kd)*self.v[ids,6:]
            +self.brake_torque(self.q[ids,7:],self.v[ids,6:]))/self.kp
        target = target.clamp(self.limits[:,0]+.06,self.limits[:,1]-.06)
        old_current = self.teacher_table[rows[:,0].long(),rows[:,1].long()]
        self.reset_teacher[ids] = (self.q[ids,7:]+(self.old_kp*(old_current-self.q[ids,7:])
            +(self.kd-self.old_kd)*self.v[ids,6:]+self.brake_torque(self.q[ids,7:],self.v[ids,6:]))/self.kp).clamp(
                self.limits[:,0]+.06,self.limits[:,1]-.06)
        self.factory_prior[ids]=0
        self.factory_prior[ids[:,None],self.q.new_tensor(IDS,dtype=torch.long)[None]] = (target-self.factory_default)/.25

    def reset(self,ids,canonical=False):
        if not getattr(self,'factory_ready',False):return super().reset(ids,canonical)
        if not len(ids):return
        self.sim.reset(ids)
        self.episode_end_frames[ids]=10**9
        self.episode_control_limits[ids]=1500
        keys=list(self.pools)
        selected=torch.tensor(keys,device=self.device)[torch.randint(len(keys),(len(ids),),device=self.device)]
        for clip in keys:
            ci = ids[selected==clip]
            if not len(ci):continue
            pools=self.pools[clip]
            for role in ('source','transition','stand'):
                mask=(self.roles[ci]<6) if role=='source' else (self.roles[ci]==(6 if role=='transition' else 7))
                group=ci[mask]
                if not len(group):continue
                if role=='source':
                    pool=pools['source']
                    eligible=pool[self.expert_resets[pool,1]<=self.source_stops[clip]-self.source_duration]
                    if len(eligible):pool=eligible
                    rows=self.expert_resets[pool[torch.randint(len(pool),(len(group),),device=self.device)]]
                    self.restore(group,rows)
                    self.episode_end_frames[group]=self.frames[group]+self.source_duration
                    self.episode_control_limits[group]=self.source_duration
                elif role=='stand':
                    stand_clip=clip if len(pools['stand']) else self.standing_clips[0]
                    pool=self.pools[stand_clip]['stand'];rows=self.expert_resets[pool[torch.randint(len(pool),(len(group),),device=self.device)]]
                    # Half initial and half terminal standing; both contain
                    # actual physical expert states. Never substitute a pose
                    # reset for a claimed successful standing rollout.
                    initial=torch.rand(len(group),device=self.device)<.5
                    pool=self.initial_standing_pool
                    rows[initial]=self.expert_resets[pool[torch.randint(len(pool),(int(initial.sum()),),device=self.device)]]
                    self.restore(group,rows)
                else:
                    # Acquisition, normal braking, and causal packet-loss return.
                    variants=[('acquire',None),('brake',None)]
                    variants += [('loss',i) for i,row in enumerate(self.meta['clips']) if row.get('generated_interruption') and row['name'].startswith(self.meta['clips'][clip]['name']+'_')]
                    choice=torch.randint(len(variants),(len(group),),device=self.device)
                    for vi,(kind,override) in enumerate(variants):
                        gi=group[choice==vi]
                        if not len(gi):continue
                        if kind=='loss':
                            fault=int(self.source_stops[override]);er=self.expert_resets
                            pool=((er[:,0]==clip)&(er[:,1]>=fault-50)&(er[:,1]<fault)).nonzero().flatten()
                            end=fault+150
                        else:
                            pool=pools[kind]
                            end=int(self.source_starts[clip] if kind=='acquire' else self.source_stops[clip])+100
                        rows=self.expert_resets[pool[torch.randint(len(pool),(len(gi),),device=self.device)]]
                        self.restore(gi,rows,override)
                        self.episode_end_frames[gi]=end
                        self.episode_control_limits[gi]=250
        self.v[ids,:2]+=.06*(torch.rand((len(ids),2),device=self.device)-.5)
        self.prior[ids]=0
        for h in self.history:h[ids]=0
        self.age[ids]=0;self.bad[ids]=False;self.episode_return[ids]=0
        self.past_proprio[ids]=self.proprioception()[ids,None]
        self.sim.forward();self.reset_count+=len(ids)

    def step(self,targets):
        prop=self.proprioception().clone()
        self.past_proprio[:,:-1]=self.past_proprio[:,1:].clone()
        self.past_proprio[:,-1]=prop
        self.factory_prior[:]=0
        self.factory_prior[:,IDS]=((targets-self.factory_default)/.25).to(self.factory_prior.dtype)
        return super().step(targets)
