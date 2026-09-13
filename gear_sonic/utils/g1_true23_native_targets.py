"""Full native target authority around the retained received-pose controller.

Factory locomotion stays a prior. New joint corrections are expressed directly
under the original benchmark PD law, not compressed through factory gains.
"""
import numpy as np
import torch
import yaml
from pathlib import Path
from gear_sonic.utils.g1_true23_task_commands import TaskCommandActor,TaskCommandController
from gear_sonic.utils.g1_true23_factory_locomotion import IDS


class NativeTargetActor(TaskCommandActor):
    def __init__(self,weights,config,contract,base_checkpoint):
        super().__init__(weights,config,contract['joint_limits'],base_checkpoint,full_body_corrections=True)
        cfg=yaml.safe_load(Path(config).read_text(encoding='utf-8'))
        values=dict(native_kp=contract['kp'],native_kd=contract['kd'],
            observation_default=contract['default_q'],factory_kp=np.asarray(cfg['joint_kp'])[IDS],
            factory_kd=np.asarray(cfg['joint_kd'])[IDS])
        for name,value in values.items():self.register_buffer(name,torch.tensor(value,dtype=torch.float32))

    def from_commands(self,x,correction,raw=None):
        if raw is None:raw=self.base.backbone(x[:,:210].reshape(-1,5,42),x[:,210:218])
        command=x[:,210:218];phase=torch.pi*torch.tanh(correction[:,23:25])
        sine,cosine=command[:,:2],command[:,2:4]
        velocity=(command[:,5:]/.2+torch.tanh(correction[:,25:])*x.new_tensor([.6,.5,3.])).clamp(
            -x.new_tensor([1.2,1.,6.]),x.new_tensor([1.2,1.,6.]))
        changed=torch.cat((sine*torch.cos(phase)+cosine*torch.sin(phase),
            cosine*torch.cos(phase)-sine*torch.sin(phase),command[:,4:5],velocity*.2),1)
        changed_raw=self.base.backbone(x[:,:210].reshape(-1,5,42),changed)
        baseline=self.default+self.base(x[:,:1542])
        legs=baseline[:,:12]+self.base.factory_scale*(changed_raw-raw)
        previous=self.default[:12]+x[:,:210].reshape(-1,5,42)[:,-1,30:42]
        alpha=1-.1*x[:,1581:1582]
        factory_target=torch.cat((previous+alpha*(legs-previous),baseline[:,12:]),1)
        # Received proprioception contains every measured joint, including arms.
        q=x[:,218:241]+self.observation_default;v=x[:,241:264]
        native_base=q+(self.factory_kp*(factory_target-q)+(self.native_kd-self.factory_kd)*v)/self.native_kp
        requested=native_base+correction[:,:23]*(self.limits[:,1]-self.limits[:,0])*.5
        clipped=requested.clamp(self.limits[:,0],self.limits[:,1])
        # Preserve the exact clipped forward value even for large proposals;
        # the zero-valued term supplies the straight-through training gradient.
        return clipped.detach()+(requested-requested.detach())-self.default

    def target(self,action):return (self.default+action).clamp(self.limits[:,0],self.limits[:,1])


class NativeTargetController(TaskCommandController):
    def __init__(self,actor,firmware,contract,*,native_preview_guard=False,native_standing_capture=False,
            native_preview_delay_substeps=0,native_preview_library=None,**options):
        super().__init__(actor,firmware,contract,**options)
        self.factory_kp=self.kp.copy();self.factory_kd=self.kd.copy()
        self.kp=np.asarray(contract['kp'],np.float64).copy();self.kd=np.asarray(contract['kd'],np.float64).copy()
        self.target_margin=0.;self.previous_native=None
        self.native_standing_capture=bool(native_standing_capture)
        self.native_standing_captured=False;self.standing_stationary_since=None
        self.native_motion_seen=False
        self.preview_guard=None
        if native_preview_guard:
            from gear_sonic.utils.g1_true23_native_preview_guard import NativePreviewGuard
            self.preview_guard=NativePreviewGuard(options['model'],contract,
                delay_substeps=native_preview_delay_substeps,library=native_preview_library)
        wrapper_file=Path(actor).with_suffix('.wrapper.json')
        if wrapper_file.exists():
            import json
            from gear_sonic.utils.g1_true23_controller_state import wrapper_contract
            if json.loads(wrapper_file.read_text())!=wrapper_contract(self):
                raise ValueError('Actor checkpoint requires different controller wrapper settings')
        reference_file=Path(actor).with_suffix('.reference.json')
        if reference_file.exists():
            import json
            from gear_sonic.utils.g1_true23_controller_state import reference_configuration
            if json.loads(reference_file.read_text())!=reference_configuration(self):
                raise ValueError('Actor checkpoint requires different standing/task reference settings')

    def prepare(self,qpos,qvel,now):
        qpos,qvel,rotation,upper=super().prepare(qpos,qvel,now)
        self.native_standing_captured=False
        if self.native_standing_capture:
            from gear_sonic.utils.g1_true23_standing_capture import standing_capture
            r={key:value[-1:] for key,value in self.last_reference.items()}
            seen,since,captured=standing_capture(r,np.asarray(qpos)[None],np.asarray(qvel)[None],
                np.asarray(self.receiver.standing_qpos)[None,7:19],np.array([self.native_motion_seen]),
                np.array([np.nan if self.standing_stationary_since is None else self.standing_stationary_since]),now)
            self.native_motion_seen=bool(seen[0])
            self.standing_stationary_since=None if np.isnan(since[0]) else float(since[0])
            capture=bool(captured[0])
            if capture:
                # Only stop the factory gait clock. All received leg, foot,
                # arm and head objectives still enter the learned controller.
                # The packet fault latch and explicit rearm rule are unchanged.
                self.velocity[:]=0.;self.policy.walking=False;self.policy.phase=0.
                self.native_standing_captured=True
        return qpos,qvel,rotation,upper

    def observation(self,qpos,qvel,now):
        if self.previous_native is not None:
            q,v=np.asarray(qpos)[7:],np.asarray(qvel)[6:]
            equivalent=q+(self.kp*(self.previous_native-q)+(self.factory_kd-self.kd)*v)/self.factory_kp
            self.policy.previous[:]=equivalent[:12]-self.policy.default[:12]
        return super().observation(qpos,qvel,now)

    def snapshot(self,now):
        from gear_sonic.utils.g1_true23_controller_state import snapshot_controller
        return snapshot_controller(self,now)

    def restore_snapshot(self,snapshot):
        from gear_sonic.utils.g1_true23_controller_state import restore_controller
        return restore_controller(self,snapshot)

    def warmup(self,qpos,qvel,now,calls=10):
        before=self.snapshot(now)
        try:
            for _ in range(calls):
                self.command(qpos,qvel,now);self.restore_snapshot(before)
        finally:self.restore_snapshot(before)

    def command(self,qpos,qvel,now):
        result=super().command(qpos,qvel,now)
        result.status['leg_target_filter_alpha']=1.
        result.status['factory_prior_filter_alpha']=.9
        result.status['native_standing_capture_enabled']=self.native_standing_capture
        result.status['native_standing_captured']=self.native_standing_captured
        result.status['native_motion_seen']=self.native_motion_seen
        result.status['native_target_actuation']=dict(original_benchmark_pd=True,target_margin_rad=0.,
            additional_limit_brake=False,original_physical_limits=True,qualified=False)
        if self.preview_guard is not None:
            target,details=self.preview_guard.apply(qpos,qvel,result.targets,previous=self.previous_native)
            from gear_sonic.utils.g1_true23_causal_controller import ControllerCommand
            result=ControllerCommand(target,dict(result.status,native_preview_guard=details))
        return result

    def commit_applied(self,qpos,qvel,target):
        self.previous_native=np.asarray(target,np.float64).copy()
        super().commit_applied(qpos,qvel,target)

    def rearm(self,now,measured_qpos):
        super().rearm(now,measured_qpos)
        self.native_motion_seen=False;self.standing_stationary_since=None
        self.native_standing_captured=False

    def import_native_history(self,control):
        """Rebuild factory memory from the independent plant's actual history."""
        super().import_native_history(control)
        if control==0:
            self.previous_native=None
            return
        c=self.receiver.contract;memory=self.receiver.history;p=self.policy
        scale=.25*c['training_effort']/c['kp']
        self.previous_native=c['default_q']+memory.prior*scale
        actions,gyro,q,dq,gravity=memory.terms
        measured=q[:,:12]+c['default_q'][:12]
        applied=c['default_q'][:12]+actions[:,:12]*scale[:12]
        equivalent=measured+(self.kp[:12]*(applied-measured)
            +(self.factory_kd[:12]-self.kd[:12])*dq[:,:12])/self.factory_kp[:12]
        previous=equivalent-p.default[:12]
        valid=min(control,4)
        if control<=4:previous[valid-1]=0.
        newest=np.concatenate((gyro*.8,gravity,measured-p.default[:12],dq[:,:12]*.05,previous),1)
        newest=np.clip(newest,-p.cfg['observation_clip'],p.cfg['observation_clip'])
        newest[valid:]=newest[valid-1]
        p.history[1:]=newest[::-1]
