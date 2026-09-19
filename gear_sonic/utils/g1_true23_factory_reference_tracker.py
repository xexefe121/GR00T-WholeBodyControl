"""Received-pose error coordinates around proven native23 factory balance.

The same global factory standing equilibrium is used for every input stream.
Received velocities are backward differences. Joint bias-force compensation is
converted to position targets; no base wrench or contact force is applied.
This is a simulation candidate, not a qualified hardware controller.
"""
from pathlib import Path
import numpy as np
import mujoco
import onnxruntime as ort
import yaml
from gear_sonic.utils.g1_true23_causal_receiver import CausalReceiver
from gear_sonic.utils.g1_true23_causal_controller import ControllerCommand


class Native23ReferenceTracker:
    def __init__(self,actor,config,equilibrium,contract,*,model,tasks,standing_qpos,now=0.,bias_compensation=True):
        cfg=self.cfg=yaml.safe_load(Path(config).read_text())
        self.default=np.asarray(cfg['default_joint_q'],np.float32)
        self.scale=np.asarray(cfg['action_scale'],np.float32)*cfg['action_scale_coeff']
        self.kp,self.kd=np.asarray(cfg['joint_kp']),np.asarray(cfg['joint_kd'])
        contract=dict(contract)
        for key in ('default_q','kp','kd','training_effort','joint_limits','native_effort','native_velocity'):contract[key]=np.asarray(contract[key])
        self.limits=contract['joint_limits']
        self.receiver=CausalReceiver(contract,model=model,tasks=tasks,standing_qpos=standing_qpos,now=now)
        self.model=model;self.actual=mujoco.MjData(model);self.virtual=mujoco.MjData(model)
        self.eq=np.asarray(equilibrium['qpos'],np.float64)
        self.eq_target=np.asarray(equilibrium['target'],np.float64)
        self.eq_rotation=np.empty(9);mujoco.mju_quat2Mat(self.eq_rotation,self.eq[3:7]);self.eq_rotation=self.eq_rotation.reshape(3,3)
        self.history=np.zeros((5,76),np.float32);self.initialized=False
        self.previous_scaled=(self.eq_target-self.default).astype(np.float32)
        self.bias_compensation=bias_compensation
        opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(actor),sess_options=opts,providers=['CPUExecutionProvider'])
        if self.session.get_inputs()[0].shape[1:]!=[5,76]:raise ValueError('native23 factory observation contract mismatch')
        self.reference_q=self.eq[7:].copy();self.feedforward=np.zeros(23)

    def receive(self,packet,task_position,task_quaternion,now):return self.receiver.receive(packet,task_position,task_quaternion,now)

    def command(self,qpos,qvel,now):
        qpos,qvel=np.asarray(qpos),np.asarray(qvel)
        if qpos.shape!=(30,) or qvel.shape!=(29,) or not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
            self.receiver.gate.latch('invalid_robot_observation',now);raise ValueError('invalid observed robot state')
        ref={k:v[-1] for k,v in self.receiver.reference(now).items()}
        r=np.empty(9);mujoco.mju_quat2Mat(r,qpos[3:7]);r=r.reshape(3,3)
        vr=self.eq_rotation@ref['root_rotation'].T@r
        virtual_q=qpos[7:]-ref['joint']+self.eq[7:]
        virtual_v=qvel[6:]-ref['joint_velocity']
        omega=qvel[3:6]-r.T@ref['root_omega']
        velocity=virtual_v.copy();velocity[[4,5,10,11]]=0
        cfg=self.cfg
        obs=np.r_[omega*cfg['observation_scale_ang_vel'],-vr[2]*cfg['observation_scale_proj_grav'],
            (virtual_q-self.default)*cfg['observation_scale_dof_pos'],velocity*cfg['observation_scale_dof_vel'],
            self.previous_scaled*cfg['observation_scale_actions'],0.].astype(np.float32)
        obs=np.clip(obs,-cfg['observation_clip'],cfg['observation_clip'])
        if not self.initialized:self.history[:]=obs;self.initialized=True
        else:self.history[-1]=obs
        action=self.session.run(['act'],{'obs':self.history[None]})[0][0]
        if action.shape!=(23,) or not np.isfinite(action).all():raise ValueError('invalid native factory output')
        virtual_target=self.default+np.clip(action,-cfg['action_clip'],cfg['action_clip'])*self.scale
        # Cancel the reference component of physical PD damping, giving the
        # original controller feedback on relative joint velocity.
        feedforward=self.kd/self.kp*ref['joint_velocity']
        if self.bias_compensation:
            self.actual.qpos[:]=qpos;self.actual.qvel[:]=qvel;mujoco.mj_forward(self.model,self.actual)
            self.virtual.qpos[:]=self.eq;self.virtual.qpos[7:]=virtual_q
            mujoco.mju_mat2Quat(self.virtual.qpos[3:7],np.ascontiguousarray(vr).ravel())
            self.virtual.qvel[:3]=qvel[:3]-ref['root_velocity']
            self.virtual.qvel[3:6]=omega;self.virtual.qvel[6:]=virtual_v
            mujoco.mj_forward(self.model,self.virtual)
            feedforward+=(self.actual.qfrc_bias[6:]-self.virtual.qfrc_bias[6:])/self.kp
        target=np.clip(ref['joint']+(virtual_target-self.eq[7:])+feedforward,self.limits[:,0]+.06,self.limits[:,1]-.06)
        self.reference_q=ref['joint'].copy();self.feedforward=feedforward
        return ControllerCommand(target,dict(mode=self.receiver.mode,epoch=self.receiver.gate.epoch,
            fault=self.receiver.gate.fault,explicit_rearm_required=self.receiver.gate.fault is not None,
            future_reference_frames=0,qualified=False,controller='native23_factory_pose_error',
            joint_bias_compensation=self.bias_compensation))

    def commit_applied(self,qpos,qvel,target):
        self.history[:-1]=self.history[1:]
        virtual_target=np.asarray(target)-self.reference_q-self.feedforward+self.eq[7:]
        self.previous_scaled=(virtual_target-self.default).astype(np.float32)
        self.receiver.commit(qpos,qvel,target)

    def rearm(self,now,measured_qpos):self.receiver.rearm(now,measured_qpos)
