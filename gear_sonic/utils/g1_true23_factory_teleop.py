"""Existing full-body packet receiver connected to the factory-derived actor.

No robot communication. Caller supplies measured state, accepts the 23-target
command and owns the physical clock. Packet loss uses the existing received-only
standing return and latched explicit-rearm gate.
"""
import numpy as np
import mujoco
from gear_sonic.utils.g1_true23_causal_controller import ControllerCommand
from gear_sonic.utils.g1_true23_causal_receiver import CausalReceiver
from gear_sonic.utils.g1_true23_factory_policy import FactoryReceivedController,IDS


class FactoryTeleopController:
    def __init__(self,actor,config,contract,*,model,tasks,**receiver_options):
        contract=dict(contract)
        for key in ('default_q','kp','kd','training_effort','joint_limits','native_effort','native_velocity'):
            contract[key]=np.asarray(contract[key])
        self.receiver=CausalReceiver(contract,model=model,tasks=tasks,**receiver_options)
        self.factory=FactoryReceivedController(actor,config,contract['joint_limits'])
        self.model=model;self.data=mujoco.MjData(model);self.contract=contract
        self.feet_ids=[model.body(side+'_ankle_roll_link').id for side in ('left','right')]
        self.task_ids=[model.body(task['target_body']).id for task in tasks]
        self.task_offsets=np.asarray([task['target_point'] for task in tasks])

    def receive(self,packet,task_position,task_quaternion,now):
        return self.receiver.receive(packet,task_position,task_quaternion,now)

    def command(self,qpos,qvel,now):
        qpos,qvel=np.asarray(qpos),np.asarray(qvel)
        if qpos.shape!=(30,) or qvel.shape!=(29,) or not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
            self.receiver.gate.latch('invalid_robot_observation',now)
            raise ValueError('invalid robot observation; no target emitted')
        ref={k:v[-1] for k,v in self.receiver.reference(now).items()}
        quaternion=np.empty(4);mujoco.mju_mat2Quat(quaternion,np.asarray(ref['root_rotation'],np.float64).ravel())
        if quaternion[0]<0:quaternion=-quaternion
        q29,v29=np.zeros(29),np.zeros(29)
        q29[IDS],v29[IDS]=ref['joint'],ref['joint_velocity']
        target=np.r_[ref['root'],quaternion[[1,2,3,0]],q29,
            ref['root_velocity']@ref['root_rotation'],ref['root_omega']@ref['root_rotation'],v29].astype(np.float32)
        self.data.qpos[:]=qpos
        mujoco.mj_kinematics(self.model,self.data)
        feet=self.data.xpos[self.feet_ids]
        tasks=self.data.xpos[self.task_ids]+np.einsum('tij,tj->ti',self.data.xmat[self.task_ids].reshape(3,3,3),self.task_offsets)
        command=self.factory.step(qpos,qvel,target,feet,tasks,ref['root'],ref['feet'],ref['tasks'])
        return ControllerCommand(command,dict(mode=self.receiver.mode,epoch=self.receiver.gate.epoch,
            fault=self.receiver.gate.fault,explicit_rearm_required=self.receiver.gate.fault is not None,
            received_samples=len(self.receiver.samples),future_reference_frames=0,qualified=False,
            controller='factory_received_native23'))

    def commit_applied(self,qpos,qvel,target):
        self.factory.commit_applied(target)
        self.receiver.commit(qpos,qvel,target)

    def rearm(self,now,measured_qpos):
        self.receiver.rearm(now,measured_qpos)
