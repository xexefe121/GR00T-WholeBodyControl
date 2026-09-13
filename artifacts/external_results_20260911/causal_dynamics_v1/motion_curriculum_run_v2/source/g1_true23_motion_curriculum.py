"""Fixed motion/transition/standing quotas and free-running source curriculum."""
import numpy as np
import torch
from gear_sonic.envs.mjlab.g1_true23_focused_dynamics import FocusedDirectBodyNative23Env


class MotionCurriculumEnv(FocusedDirectBodyNative23Env):
    STAGE_CONTROLS=(100,200,400,None)

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.count<64 or self.count%8:raise ValueError('curriculum count must be >=64 and divisible by8')
        self.stage=0
        self.roles=torch.full((self.count,),2,device=self.device,dtype=torch.long)
        self.roles[:self.count*6//8]=0
        self.roles[self.count*6//8:self.count*7//8]=1
        self.episode_end_frames=torch.full_like(self.frames,10**9)
        self.episode_control_limits=torch.full_like(self.frames,1500)
        row=self.meta['clips'][self.focus_id]
        starts,stops=row['source_start']+11,row['source_stop']+11
        same=self.expert_resets[:,0]==self.focus_id
        frame=self.expert_resets[:,1]
        self.acquisition_pool=(same&(frame>=starts-100)&(frame<starts)).nonzero().flatten()
        # Normal braking starts while the source is still moving, then crosses
        # into the return. Resetting only after source_stop skips this handoff.
        self.return_pool=(same&(frame>=stops-100)&(frame<stops)).nonzero().flatten()
        # Verified actual quiet states, including their prior applied commands.
        self.standing_pool=(same&(frame>=stops+100)&
            (self.expert_resets[:,32:61].abs().amax(-1)<.5)).nonzero().flatten()
        if not all(len(p) for p in (self.acquisition_pool,self.return_pool,self.standing_pool)):
            raise ValueError('curriculum needs actual acquisition, return and quiet states')
        self.quota_ready=True
        self.reset(torch.arange(self.count,device=self.device))

    @property
    def stage_controls(self):
        return self.STAGE_CONTROLS[self.stage] or int(self.source_stops[self.focus_id]-self.source_starts[self.focus_id])

    def draw(self,pool,count):
        return self.expert_resets[pool[torch.randint(len(pool),(count,),device=self.device)]]

    def reset(self,ids,canonical=False):
        if not getattr(self,'quota_ready',False):return super().reset(ids,canonical)
        if not len(ids):return
        self.sim.reset(ids)
        start=int(self.source_starts[self.focus_id]);stop=int(self.source_stops[self.focus_id])
        self.episode_end_frames[ids]=10**9;self.episode_control_limits[ids]=1500
        source=ids[self.roles[ids]==0]
        if len(source):
            eligible=self.focus_motion_pool[self.expert_resets[self.focus_motion_pool,1]<=stop-self.stage_controls]
            self.restore_rows(source,self.draw(eligible,len(source)))
            self.episode_end_frames[source]=self.frames[source]+self.stage_controls
            self.episode_control_limits[source]=self.stage_controls
        transition=ids[self.roles[ids]==1]
        if len(transition):
            choice=torch.randint(2+len(self.focus_recovery_pools),(len(transition),),device=self.device)
            for kind in range(2+len(self.focus_recovery_pools)):
                selected=transition[choice==kind]
                if not len(selected):continue
                if kind<2:
                    pool=self.acquisition_pool if kind==0 else self.return_pool
                    self.restore_rows(selected,self.draw(pool,len(selected)))
                    # Both transitions must include states on each side of
                    # their boundary, without changing the75% source quota.
                    self.episode_end_frames[selected]=start+100 if kind==0 else stop+100
                else:
                    clip,pool=self.focus_recovery_pools[kind-2]
                    self.restore_rows(selected,self.draw(pool,len(selected)),clip)
                    self.episode_end_frames[selected]=self.fault_frames[clip]+100
                self.episode_control_limits[selected]=200
        standing=ids[self.roles[ids]==2]
        if len(standing):self.restore_rows(standing,self.draw(self.standing_pool,len(standing)))
        self.v[ids,:2]+=.06*(torch.rand((len(ids),2),device=self.device)-.5)
        clip=self.clips[ids];frame=torch.minimum(self.frames[ids],self.lengths[clip]-1)
        self.motion_fraction[ids]=self.reference_blend[clip,frame]
        self.age[ids]=0;self.bad[ids]=False;self.episode_return[ids]=0
        self.fault[ids]=False;self.anchor[ids]=0
        self.sim.forward();self.reset_count+=len(ids)

    def noise_radians(self,moving=.06,standing=.03):
        return torch.where(self.roles==0,moving,standing)[:,None]

    @torch.no_grad()
    def evaluate_segments(self,actor):
        """64 fixed actual starts; deterministic policy, no runtime readiness claim.

        This explicitly resets training environments at a PPO update boundary.
        Evaluation masks end at each first failure or segment cutoff. Subsequent
        automatic resets cannot improve that start's score.
        """
        count=64;ids=torch.arange(count,device=self.device)
        stop=int(self.source_stops[self.focus_id]);length=self.stage_controls
        pool=self.focus_motion_pool[self.expert_resets[self.focus_motion_pool,1]<=stop-length]
        selected=pool[torch.linspace(0,len(pool)-1,count,device=self.device).long()]
        roles=self.roles.clone();self.roles[:count]=0
        self.sim.reset(ids);self.restore_rows(ids,self.expert_resets[selected])
        self.v[ids,0]+=torch.tensor([-.03,0.,.03,0.],device=self.device).repeat(16)
        self.age[ids]=0;self.bad[ids]=False;self.episode_return[ids]=0
        self.episode_end_frames[ids]=self.frames[ids]+length;self.episode_control_limits[ids]=length
        self.motion_fraction[ids]=self.reference_blend[self.clips[ids],self.frames[ids]]
        self.sim.forward();alive=torch.ones(count,device=self.device,dtype=torch.bool)
        failed=torch.zeros_like(alive);completed=torch.zeros_like(alive)
        values={key:[] for key in ('root_error','yaw_error','foot_errors','task_errors','leg_error')}
        masks=[]
        for _ in range(length):
            _,_,done,info=self.step(actor.target(actor(self.observe())))
            masks.append(alive.cpu().numpy().copy())
            for key in values:values[key].append(info[key][:count].cpu().numpy())
            failed|=alive&(info['failed'][:count]|info['tracking_failed'][:count])
            completed|=alive&info['truncated'][:count]
            alive&=~done[:count]
            if not alive.any():break
        masks=np.stack(masks);values={k:np.stack(v) for k,v in values.items()}
        passed=[];scores=[]
        for i in range(count):
            x={k:v[masks[:,i],i] for k,v in values.items()}
            root=float(np.percentile(x['root_error'],95));yaw=float(np.percentile(x['yaw_error'],95))
            feet=np.percentile(x['foot_errors'],95,axis=0);tasks=np.percentile(x['task_errors'],95,axis=0)
            legs=float(np.sqrt(np.mean(x['leg_error']**2)))
            okay=bool(completed[i] and not failed[i] and len(x['root_error'])==length and root<=.2 and
                yaw<=np.deg2rad(15) and np.all(feet<=.12) and np.all(tasks<=[.15,.15,.1]) and legs<=.15)
            passed.append(okay);scores.append(dict(root_p95_m=root,yaw_p95_deg=float(np.rad2deg(yaw)),
                feet_p95_m=feet.tolist(),tasks_p95_m=tasks.tolist(),leg_rmse_rad=legs,controls=len(x['root_error']),passed=okay))
        result=dict(stage=self.stage,segment_seconds=length*.02,starts=count,passed=sum(passed),pass_fraction=float(np.mean(passed)),
            simulator='MuJoCo-Warp3.5; training reference metrics; native full-lifecycle evaluator remains authoritative',
            cases=scores,full_motion_qualification=False)
        if result['pass_fraction']>=.9 and self.stage<len(self.STAGE_CONTROLS)-1:self.stage+=1
        result['next_stage']=self.stage
        self.roles.copy_(roles);self.reset(torch.arange(self.count,device=self.device))
        return result
