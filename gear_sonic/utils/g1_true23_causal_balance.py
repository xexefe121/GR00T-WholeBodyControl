"""Experimental full-range motion takeover with retained native23 balance.

BFM uses one fixed neutral pose plus measured XY/yaw/velocity feedback. Motion
selection uses only the last received pose and its measured past. No timeline
phase, per-motion gain schedule or future body pose is available here.
"""
import numpy as np
import torch
from gear_sonic.utils.g1_true23_neutral_balance import NeutralBalance
from gear_sonic.utils.g1_true23_causal_controller import CausalController,ControllerCommand


class BalancedCausalController(CausalController):
    def __init__(self,actor,contract,*,weights,neutral_motion,**options):
        super().__init__(actor,contract,**options)
        torch.set_num_threads(1)
        self.balance=NeutralBalance(weights,contract,neutral_motion)
        self.neutral=np.asarray(neutral_motion['joint_pos'][0])
        self.alpha=0.;self.fault_anchor=None;self.neutral_received=None

    def command(self,qpos,qvel,now):
        candidate=super().command(qpos,qvel,now)
        sample=self.receiver.samples[-1]
        if self.neutral_received is None:self.neutral_received=np.asarray(self.receiver.samples[0]['joint_pos']).copy()
        if len(self.receiver.samples)>1:
            previous=self.receiver.samples[-2]
            speed=np.max(np.abs(sample['joint_pos']-previous['joint_pos']))/.02
            root_speed=np.linalg.norm(sample['body_pos_w'][0]-previous['body_pos_w'][0])/.02
            root_turn=2*np.arccos(np.clip(abs(np.dot(sample['body_quat_w'][0],previous['body_quat_w'][0])),0,1))/.02
        else:speed=root_speed=root_turn=0.
        neutral_distance=min(np.max(np.abs(sample['joint_pos']-self.neutral_received)),np.max(np.abs(sample['joint_pos']-self.neutral)))
        moving=neutral_distance>.04 or speed>.10 or root_speed>.025 or root_turn>.10
        fault=self.receiver.gate.fault is not None
        desired=1. if moving and not fault else 0.
        # Full-range targets blend over 0.5s. This is not a small residual cap.
        self.alpha=float(np.clip(self.alpha+np.clip(desired-self.alpha,-.04,.04),0.,1.))
        qw,qx,qy,qz=qpos[3:7]
        actual_yaw=np.arctan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))
        if fault:
            if self.fault_anchor is None:self.fault_anchor=np.r_[qpos[:2],actual_yaw]
            anchor=self.fault_anchor
        else:
            qw,qx,qy,qz=sample['body_quat_w'][0]
            reference_yaw=np.arctan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))
            anchor=np.r_[sample['body_pos_w'][0,:2],reference_yaw]
        values=(qpos,qvel,self.receiver.history.prior,self.receiver.history.vector(),anchor)
        q,v,prior,history,anchor=[torch.as_tensor(x,dtype=torch.float32)[None] for x in values]
        balance=self.balance.target(q,v,prior,history,anchor)[0].numpy().astype(float)
        targets=np.clip((1-self.alpha)*balance+self.alpha*candidate.targets,self.limits[:,0],self.limits[:,1])
        status={**candidate.status,'motion_fraction':self.alpha,'balance_supervisor':True,
            'mode':'fault_stopping' if fault else 'full_body_motion' if self.alpha==1 else 'motion_takeover' if self.alpha>0 else 'standing_balance'}
        return ControllerCommand(targets,status)

    def rearm(self,now,measured_qpos):
        super().rearm(now,measured_qpos);self.fault_anchor=None;self.neutral_received=None
