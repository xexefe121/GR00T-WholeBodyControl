"""Reusable received-body-reference -> 23-target controller interface.

The caller owns the physical clock and must acknowledge the target actually
applied once per control tick. This interface never advances physics and does
not turn candidate policies into qualified controllers.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import onnxruntime as ort
from gear_sonic.utils.g1_true23_causal_receiver import CausalReceiver


@dataclass(frozen=True)
class ControllerCommand:
    targets: np.ndarray
    status: dict


class CausalController:
    def __init__(self,actor,contract,**receiver_options):
        actor=Path(actor)
        with np.load(actor.with_suffix('.normalization.npz'),allow_pickle=False) as z:
            self.default=z['default'].astype(float)
            self.span=z['span'].astype(float)
            self.limits=z['limits'].copy()
        np.testing.assert_allclose(self.limits,contract['joint_limits'],rtol=1e-7,atol=1e-7)
        np.testing.assert_allclose(self.default,contract['default_q'],rtol=1e-7,atol=1e-7)
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
        options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        self.session=ort.InferenceSession(str(actor),sess_options=options,providers=['CPUExecutionProvider'])
        self.receiver=CausalReceiver(contract,**receiver_options)
        self.body_goal=None
        width=self.session.get_inputs()[0].shape[-1]
        if width==1749:
            from gear_sonic.utils.g1_true23_direct_body_goal import ReceivedBodyGoal
            self.body_goal=ReceivedBodyGoal(contract)
        elif width!=1323:raise ValueError('unsupported causal controller input width')

    def receive(self,packet,task_position,task_quaternion,now):
        return self.receiver.receive(packet,task_position,task_quaternion,now)

    def command(self,qpos,qvel,now):
        qpos,qvel=np.asarray(qpos),np.asarray(qvel)
        if qpos.shape!=(30,) or qvel.shape!=(29,) or not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
            self.receiver.gate.latch('invalid_robot_observation',now)
            raise ValueError('invalid robot observation; no target emitted')
        features=self.receiver.features(qpos,qvel,now)
        if self.body_goal is not None:features=self.body_goal.features(features,self.receiver)
        raw=self.session.run(None,{'features':features[None]})[0][0]
        if raw.shape!=(23,) or not np.isfinite(raw).all():
            self.receiver.gate.latch('invalid_policy_output',now)
            raise ValueError('invalid policy output; no target emitted')
        targets=np.clip(self.default+self.span*raw.astype(float),self.limits[:,0],self.limits[:,1])
        return ControllerCommand(targets,dict(mode=self.receiver.mode,epoch=self.receiver.gate.epoch,
            fault=self.receiver.gate.fault,explicit_rearm_required=self.receiver.gate.fault is not None,
            received_samples=len(self.receiver.samples),future_reference_frames=0,qualified=False))

    def commit_applied(self,qpos,qvel,target):
        self.receiver.commit(qpos,qvel,target)

    def rearm(self,now,measured_qpos):
        self.receiver.rearm(now,measured_qpos)
