"""Controlled legacy/replayed-memory experiment under the same native wrapper."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
import mujoco
from gear_sonic.envs.mjlab.g1_true23_native_target_dynamics import NativeTargetEnv
from gear_sonic.envs.mjlab.g1_true23_factory_dynamics import FactoryNative23Env
from gear_sonic.utils.g1_true23_batch_preview import BatchNativePreview
from gear_sonic.utils.g1_true23_standing_capture import standing_capture
from gear_sonic.utils.g1_true23_factory_policy import IDS


class MatchedMemoryNative23Env(NativeTargetEnv):
    def __init__(self,*args,memory_bank,reset_memory,preview_library,experiment_seed=0,**kwargs):
        self.memory_ready=False
        kwargs['command_delay_substeps']=3
        super().__init__(*args,**kwargs)
        if self.device!='cpu':raise ValueError('Matched native memory experiment currently requires CPU policy tensors')
        self.sim.data.ctrl=self.sim.data.ctrl.double()
        if reset_memory not in ('legacy','replayed'):raise ValueError('Unknown controller-memory arm')
        self.reset_memory=reset_memory;self.memory_bank=Path(memory_bank)
        self.memory_manifest=json.loads((self.memory_bank/'manifest.json').read_text())
        with (self.bank/'expert.npz').open('rb') as f:expert_hash=hashlib.file_digest(f,'sha256').hexdigest()
        if expert_hash!=self.memory_manifest['expert_sha256']:raise ValueError('Memory bank belongs to another expert dataset')
        with np.load(self.memory_bank/'memory.npz') as z:self.memory={k:z[k].copy() for k in z.files}
        np.testing.assert_array_equal(self.expert_resets.numpy(),self.memory['expert_rows'].astype(np.float32))
        self.reset_rng=[np.random.default_rng(experiment_seed*1000003+i) for i in range(self.count)]
        self.delay_rng=np.random.default_rng(experiment_seed+700001)
        self.command_delay_substeps=6
        self.capture_seen=np.zeros(self.count,bool);self.capture_since=np.full(self.count,np.nan)
        self.time_offset=np.zeros(self.count);self.history_valid=np.zeros(self.count,bool)
        self.last_reset_row=np.full(self.count,-1,dtype=int)
        self.native_last_applied=np.zeros((self.count,23));self.native_held=np.zeros((self.count,23))
        self.preview=BatchNativePreview(self.sim.model,self.c,self.count,preview_library,delay=6,workers=8)
        self.standing_legs=np.stack([np.asarray(json.loads((Path(args[0])/r['name'].split('_loss')[0]/'timeline.json').read_text())['configured_standing_qpos'])[7:19]
            if not r.get('generated_interruption') else np.asarray(self.c['default_q'])[:12] for r in self.meta['clips']])
        # Generated interruption names need not be filesystem clip names.
        for i,r in enumerate(self.meta['clips']):
            if r.get('generated_interruption'):
                base=next(j for j,s in enumerate(self.meta['clips']) if not s.get('generated_interruption') and r['name'].startswith(s['name']+'_'))
                self.standing_legs[i]=self.standing_legs[base]
        self.pool_arrays={clip:{k:v.numpy() for k,v in pools.items()} for clip,pools in self.pools.items()}
        self.initial_pool=self.initial_standing_pool.numpy()
        self.memory_ready=True;self.reset(torch.arange(self.count))

    def reset(self,ids,canonical=False):
        if not self.memory_ready:return super().reset(ids,canonical)
        ids=np.asarray(ids,dtype=int)
        if not len(ids):return
        keys=list(self.pools);rows=[];overrides=[];deltas=[];phases=[];canonical_flags=[]
        ends=[];durations=[]
        for index in ids:
            rng=self.reset_rng[index];role=int(self.roles[index]);clip=keys[int(rng.integers(len(keys)))];pool=self.pool_arrays[clip]
            is_canonical=2<=role<6
            if is_canonical:
                ci=int(self.canonical_assignment[index]);row=-1;end=10**9;duration=int(self.totals[ci])+1500
            elif role<2:
                eligible=pool['source'][self.memory['expert_rows'][pool['source'],1]<=int(self.source_stops[clip])-self.source_duration]
                row=int(rng.choice(eligible if len(eligible) else pool['source']));ci=clip
                end=int(self.memory['expert_rows'][row,1])+self.source_duration;duration=self.source_duration
            elif role==7:
                stand_clip=clip if len(pool['stand']) else self.standing_clips[0]
                eligible=self.initial_pool if rng.random()<.5 else self.pool_arrays[stand_clip]['stand']
                row=int(rng.choice(eligible));ci=int(self.memory['expert_rows'][row,0]);end=10**9;duration=1500
            else:
                variants=[('acquire',None),('brake',None)]+[('loss',i) for i,r in enumerate(self.meta['clips'])
                    if r.get('generated_interruption') and r['name'].startswith(self.meta['clips'][clip]['name']+'_')]
                kind,override=variants[int(rng.integers(len(variants)))]
                ci=clip if override is None else override
                if kind=='loss':
                    fault=int(self.source_stops[ci]);er=self.memory['expert_rows']
                    eligible=np.flatnonzero((er[:,0]==clip)&(er[:,1]>=fault-50)&(er[:,1]<fault));end=fault+150
                else:eligible=pool[kind];end=int(self.source_starts[clip] if kind=='acquire' else self.source_stops[clip])+100
                row=int(rng.choice(eligible));duration=250
            rows.append(row);overrides.append(ci);canonical_flags.append(is_canonical)
            deltas.append(rng.uniform(-.03,.03,size=2));phases.append(float(rng.random()))
            ends.append(end);durations.append(duration)
        self.restore_starts(ids,np.asarray(rows),np.asarray(overrides),np.asarray(deltas),np.asarray(phases),
            np.asarray(canonical_flags),np.asarray(ends),np.asarray(durations))

    def restore_starts(self,ids,rows,clips,velocity_delta=None,legacy_phase=None,canonical=None,ends=None,durations=None):
        """Explicit starts also used by paired fixed physical evaluations."""
        ids=np.asarray(ids,dtype=int);rows=np.asarray(rows,dtype=int);clips=np.asarray(clips,dtype=int);n=len(ids)
        canonical=rows<0 if canonical is None else canonical
        velocity_delta=np.zeros((n,2)) if velocity_delta is None else velocity_delta
        legacy_phase=np.zeros(n) if legacy_phase is None else legacy_phase
        states=np.empty((n,mujoco.mj_stateSize(self.sim.model,self.sim.state_spec)));warnings=np.zeros((n,8,2),dtype=np.int32)
        d=self.sim.state_scratch
        for j,(row,ci) in enumerate(zip(rows,clips)):
            if row<0:
                mujoco.mj_resetData(self.sim.model,d);d.qpos[:]=self.states[ci,10,:30].numpy();d.qvel[:]=self.states[ci,10,30:].numpy()
                mujoco.mj_forward(self.sim.model,d)
                mujoco.mj_getState(self.sim.model,d,states[j],self.sim.state_spec)
            else:states[j]=self.memory['integration'][row];warnings[j]=self.memory['warning'][row]
            mujoco.mj_setState(self.sim.model,d,states[j],self.sim.state_spec);d.qvel[:2]+=velocity_delta[j]
            mujoco.mj_getState(self.sim.model,d,states[j],self.sim.state_spec)
        self.sim.import_integration(ids,states,warnings)
        self.clips[ids]=torch.as_tensor(clips);self.canonical_world[ids]=torch.as_tensor(canonical)
        self.frames[ids]=torch.as_tensor([11 if row<0 else int(self.memory['expert_rows'][row,1]) for row in rows])
        self.episode_end_frames[ids]=torch.as_tensor(ends if ends is not None else self.frames[ids].numpy()+100)
        self.episode_control_limits[ids]=torch.as_tensor(durations if durations is not None else np.full(n,100))
        self.age[ids]=0;self.bad[ids]=False;self.episode_return[ids]=0
        self.prior[ids]=0
        for history in self.history:history[ids]=0
        self.factory_prior[ids]=0;self.command_velocity[ids]=0
        self.loco_walking[ids]=False;self.loco_phase[ids]=torch.as_tensor(legacy_phase,dtype=self.loco_phase.dtype)
        self.capture_seen[ids]=False;self.capture_since[ids]=np.nan
        self.history_valid[ids]=~canonical;self.last_reset_row[ids]=rows
        self.reset_teacher_valid[ids]=~torch.as_tensor(canonical)
        for j,(index,row) in enumerate(zip(ids,rows)):
            if row<0:
                previous=self.c['default_q'];self.loco_phase[index]=0;self.time_offset[index]=-.22
                self.reset_teacher[index]=0
            else:
                previous=self.memory['last_applied'][row];self.reset_teacher[index]=self.expert_targets[row]
                self.time_offset[index]=float(self.memory['logical_time'][row])-.02*int(self.frames[index])
            self.last_applied[index]=torch.as_tensor(previous,dtype=self.last_applied.dtype)
            self.native_last_applied[index]=previous
            self.loco_previous[index]=self.last_applied[index,:12]-self.factory_default[:12]
            self.factory_prior[index,IDS]=(self.last_applied[index]-self.factory_default)/.25
        self.loco_history[ids]=self.native_prop()[ids,None]
        if self.reset_memory=='replayed':
            for index,row in zip(ids,rows):
                if row<0:continue
                m=self.memory
                self.prior[index]=torch.as_tensor(m['prior'][row]);offset=0
                for history in self.history:
                    width=history[index].numel();history[index]=torch.as_tensor(m['history'][row,offset:offset+width]).reshape_as(history[index]);offset+=width
                self.loco_history[index]=torch.as_tensor(m['loco_history'][row])
                self.command_velocity[index]=torch.as_tensor(m['velocity'][row]);self.loco_phase[index]=float(m['phase'][row])
                self.loco_walking[index]=bool(m['walking'][row]);self.history_valid[index]=bool(m['valid'][row])
                self.capture_seen[index]=bool(m['motion_seen'][row]);self.capture_since[index]=float(m['stationary_since'][row])
        self.past_proprio[ids]=self.proprioception()[ids,None]
        self.held_for_delay[ids]=self.last_applied[ids];self.delay_steps[ids]=0
        self.native_held[ids]=self.native_last_applied[ids]
        self.reset_count+=len(ids)

    def native_prop(self):
        prop=super().native_prop()
        if self.memory_ready:
            prop[:,30:]=torch.where(torch.from_numpy(self.history_valid)[:,None],prop[:,30:],0.)
        return prop

    def control_torque(self,targets):
        if not self.memory_ready:return super().control_torque(targets)
        applied=np.where((self.control_substep<self.delay_steps.numpy())[:,None],self.native_held,self.native_last_applied)
        self.control_substep+=1
        q,v=self.sim.native['qpos'][:,7:],self.sim.native['qvel'][:,6:]
        effort=np.asarray(self.c['native_effort'])
        torque=np.clip(self.c['kp']*(applied-q)-self.c['kd']*v,-effort,effort)
        return torch.from_numpy(torque)

    def capture_proposal(self):
        idx=torch.minimum(self.frames,self.lengths[self.clips]-1)
        r={k:value[self.clips,idx].numpy() for k,value in self.references.items()}
        return standing_capture(r,self.q.numpy(),self.v.numpy(),self.standing_legs[self.clips.numpy()],
            self.capture_seen,self.capture_since,self.time_offset+self.frames.numpy()*.02)

    def control_fields(self):
        x,prop,velocity,phase,walking=super().control_fields()
        if not self.memory_ready:return x,prop,velocity,phase,walking
        _,_,captured=self.capture_proposal();mask=torch.from_numpy(captured)
        velocity=velocity.clone();phase=phase.clone();walking=walking.clone();x=x.clone()
        velocity[mask]=0;phase[mask]=0;walking[mask]=False
        x[mask,210:218]=x.new_tensor([0,0,1,1,-.6,0,0,0])
        return x,prop,velocity,phase,walking

    def observe(self):
        x=super().observe()
        if self.memory_ready:x[:,1581]=torch.from_numpy(self.history_valid).to(x.dtype)
        return x

    def step(self,targets):
        if not self.memory_ready:return super().step(targets)
        requested=targets.detach().numpy().copy()
        filtered=self.preview.apply(self.sim.native['qpos'],self.sim.native['qvel'],requested,
            self.native_last_applied,self.history_valid)
        targets=torch.as_tensor(filtered)
        self.requested_targets=requested;self.filtered_targets=filtered
        _,prop,velocity,phase,walking=self.control_fields()
        self.capture_seen,self.capture_since,_=self.capture_proposal()
        self.held_for_delay.copy_(self.last_applied);self.last_applied.copy_(targets)
        self.native_held[:]=self.native_last_applied;self.native_last_applied[:]=filtered
        self.control_substep=0
        self.delay_steps[:]=torch.from_numpy(self.delay_rng.integers(0,7,size=self.count))
        self.delay_steps[~torch.from_numpy(self.history_valid)]=0
        self.loco_history[:,:-1]=self.loco_history[:,1:].clone();self.loco_history[:,-1]=prop
        self.command_velocity[:]=velocity;self.loco_phase[:]=phase;self.loco_walking[:]=walking
        self.loco_previous[:]=targets[:,:12]-self.factory_default[:12]
        self.history_valid[:]=True
        return FactoryNative23Env.step(self,targets)
