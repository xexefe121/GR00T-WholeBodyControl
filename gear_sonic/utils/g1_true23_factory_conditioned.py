"""Full-range pose conditioning around the preserved native23 balance network."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
import yaml
from gear_sonic.utils.g1_true23_factory_native import Native23FactoryNetwork

NATIVE_CONDITIONED_WIDTH=1704  # actual native history380 + received1323 + causal closure blend1


class NativeFactoryConditionedActor(nn.Module):
    def __init__(self,weights_path,config,limits):
        super().__init__()
        cfg=yaml.safe_load(Path(config).read_text())
        self.backbone=Native23FactoryNetwork(weights_path)
        # Preserve the independently proven balance feedback. The new command
        # head can use every native joint's full legal interval.
        for p in self.backbone.parameters():p.requires_grad_(False)
        self.goal_head=nn.Sequential(nn.Linear(1323,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,23))
        nn.init.zeros_(self.goal_head[-1].weight);nn.init.zeros_(self.goal_head[-1].bias)
        self.register_buffer('default',torch.tensor(cfg['default_joint_q'],dtype=torch.float32))
        self.register_buffer('factory_scale',torch.tensor(cfg['action_scale'],dtype=torch.float32)*cfg['action_scale_coeff'])
        self.register_buffer('limits',torch.tensor(limits,dtype=torch.float32))
        self.register_buffer('span',torch.ones(23))
        self.register_buffer('goal_mean',torch.zeros(1323));self.register_buffer('goal_scale',torch.ones(1323))

    def forward(self,x):
        native=x[:,:380].reshape(-1,5,76);features=x[:,380:1703];alpha=x[:,1703]
        base=self.default+self.factory_scale*self.backbone(native)
        delta=self.goal_head(((features-self.goal_mean)/self.goal_scale).clamp(-20,20))
        # Preserve neutral leg balance while position/yaw are already aligned.
        # Static crouches or changed hand/head intent remain active through the
        # same received geometric closure test as the existing teleop receiver.
        root_error=features[:,102:104].norm(dim=-1)
        yaw=torch.atan2(features[:,107],features[:,105]).abs()
        aligned=(root_error<.03)&(yaw<.06)
        leg_gate=torch.where(aligned,alpha,torch.ones_like(alpha)).clamp(0,1)
        mask=torch.cat((leg_gate[:,None].expand(-1,13),torch.ones_like(delta[:,13:])),1)
        requested=base+delta*mask*(self.limits[:,1]-self.limits[:,0])*.5
        target=requested.clamp(self.limits[:,0]+.06,self.limits[:,1]-.06)
        # Commands are bounded; physical robot state is never projected.
        target=requested+(target-requested).detach()
        return target-self.default

    def target(self,actions):return (self.default+actions).clamp(self.limits[:,0]+.06,self.limits[:,1]-.06)


class NativeFactoryConditionedController:
    def __init__(self,actor,config,contract,*,model,tasks,standing_qpos,now=0.):
        import onnxruntime as ort
        from gear_sonic.utils.g1_true23_causal_receiver import CausalReceiver
        from gear_sonic.utils.g1_true23_direct_body_goal import ReceivedBodyGoal
        cfg=self.cfg=yaml.safe_load(Path(config).read_text())
        self.default=np.asarray(cfg['default_joint_q'],np.float32)
        self.kp,self.kd=np.asarray(cfg['joint_kp']),np.asarray(cfg['joint_kd'])
        c=dict(contract)
        for key in ('default_q','kp','kd','training_effort','joint_limits','native_effort','native_velocity'):c[key]=np.asarray(c[key])
        self.limits=c['joint_limits'];self.receiver=CausalReceiver(c,model=model,tasks=tasks,standing_qpos=standing_qpos,now=now)
        self.body_goal=ReceivedBodyGoal(c,task_closure_stand=True)
        self.history=np.zeros((5,76),np.float32);self.previous=np.zeros(23,np.float32);self.initialized=False
        opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(actor),sess_options=opts,providers=['CPUExecutionProvider'])
        if self.session.get_inputs()[0].shape[-1]!=NATIVE_CONDITIONED_WIDTH:raise ValueError('native conditioned observation width mismatch')

    def receive(self,packet,task_position,task_quaternion,now):return self.receiver.receive(packet,task_position,task_quaternion,now)

    def command(self,qpos,qvel,now):
        import mujoco
        from gear_sonic.utils.g1_true23_causal_controller import ControllerCommand
        qpos,qvel=np.asarray(qpos),np.asarray(qvel)
        if qpos.shape!=(30,) or qvel.shape!=(29,) or not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
            self.receiver.gate.latch('invalid_robot_observation',now);raise ValueError('invalid measured state')
        features=self.receiver.features(qpos,qvel,now)
        extended=self.body_goal.features(features,self.receiver)
        r=np.empty(9);mujoco.mju_quat2Mat(r,qpos[3:7])
        velocity=qvel[6:].copy();velocity[[4,5,10,11]]=0
        cfg=self.cfg
        obs=np.r_[qvel[3:6]*cfg['observation_scale_ang_vel'],-r.reshape(3,3)[2]*cfg['observation_scale_proj_grav'],
            (qpos[7:]-self.default)*cfg['observation_scale_dof_pos'],velocity*cfg['observation_scale_dof_vel'],
            self.previous*cfg['observation_scale_actions'],0.].astype(np.float32)
        obs=np.clip(obs,-cfg['observation_clip'],cfg['observation_clip'])
        if not self.initialized:self.history[:]=obs;self.initialized=True
        else:self.history[-1]=obs
        x=np.r_[self.history.ravel(),features,extended[-1]].astype(np.float32)
        raw=self.session.run(['normalized_target'],{'features':x[None]})[0][0]
        if raw.shape!=(23,) or not np.isfinite(raw).all():raise ValueError('invalid native conditioned target')
        target=np.clip(self.default+raw,self.limits[:,0]+.06,self.limits[:,1]-.06)
        return ControllerCommand(target,dict(mode=self.receiver.mode,epoch=self.receiver.gate.epoch,
            fault=self.receiver.gate.fault,explicit_rearm_required=self.receiver.gate.fault is not None,
            future_reference_frames=0,qualified=False,controller='native23_factory_conditioned',motion_fraction=float(extended[-1])))

    def commit_applied(self,qpos,qvel,target):
        self.history[:-1]=self.history[1:]
        self.previous=(np.asarray(target)-self.default).astype(np.float32)
        self.receiver.commit(qpos,qvel,target)

    def import_native_history(self,control):
        """Reconstruct native memory from the independent plant's actual history.

        The plant supplies four newest-first boundary observations and the
        command actually applied at each boundary, even when inference was late.
        Never advance memory from an uncommitted controller proposal.
        """
        if control==0:return
        memory=self.receiver.history
        c=self.receiver.contract
        old_default=np.asarray(c['default_q'])
        old_scale=.25*np.asarray(c['training_effort'])/np.asarray(c['kp'])
        actions,gyro,position,velocity,gravity=memory.terms
        previous=old_default+actions*old_scale-self.default
        # Initial native policy has no previous action. Earlier missing history
        # repeats this first measured observation, as in the verified MNN runner.
        valid=min(control,4)
        if control<=4:previous[valid-1]=0
        dq=velocity.copy();dq[:,[4,5,10,11]]=0
        newest=np.concatenate((gyro,gravity,position+old_default-self.default,
            dq*self.cfg['observation_scale_dof_vel'],previous,np.zeros((4,1))),axis=1)
        newest[valid:]=newest[valid-1]
        self.history[:4]=newest[::-1]
        self.previous[:]=old_default+memory.prior*old_scale-self.default
        self.initialized=True

    def rearm(self,now,measured_qpos):self.receiver.rearm(now,measured_qpos)
