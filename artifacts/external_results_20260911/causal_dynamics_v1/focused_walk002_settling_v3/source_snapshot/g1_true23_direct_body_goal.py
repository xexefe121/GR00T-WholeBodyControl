"""Current received-body BFM encoding; one actor, no future reference window."""
import numpy as np
import torch
import torch.nn.functional as F
from gear_sonic.utils.g1_true23_single_policy import SinglePolicy
from gear_sonic.utils.g1_true23_bfmzero_inference import reference_features
from gear_sonic.utils.g1_true23_received_features import backward_velocity,backward_omega

INPUT_SIZE=1749


def causal_body_features(motion,contract):
    causal=dict(motion)
    causal['joint_vel']=backward_velocity(motion['joint_pos'])
    causal['body_lin_vel_w']=backward_velocity(motion['body_pos_w'])
    causal['body_ang_vel_w']=backward_omega(motion['body_quat_w'])
    state,priv=reference_features(causal,contract)
    return np.concatenate((state,priv),-1).astype(np.float32)


class ReceivedBodyGoal:
    """Adds current encoded pose and a past-only goal blend to owned samples."""
    def __init__(self,contract):
        self.contract=contract;self.alpha=0.;self.epoch=None;self.first=None

    def features(self,base,receiver):
        if receiver.gate.epoch!=self.epoch:
            self.epoch=receiver.gate.epoch;self.alpha=0.;self.first=None
        if self.first is None:self.first=np.array(receiver.samples[0]['joint_pos'],copy=True)
        samples=list(receiver.samples)[-2:]
        motion={k:np.stack([s[k] for s in samples]) for k in samples[-1]}
        encoded=causal_body_features(motion,self.contract)[-1]
        result=np.r_[base,encoded,self.alpha].astype(np.float32)
        joint=base[56:79]
        near=min(np.max(np.abs(joint-self.first)),np.max(np.abs(joint-self.contract['default_q'])))<=.04
        moving=(not near) or np.max(np.abs(base[79:102]))>.10 or np.linalg.norm(base[111:114])>.025 or np.linalg.norm(base[114:117])>.10
        desired=float(moving and receiver.gate.fault is None)
        # The current command uses the previous blend. Advance only once per
        # control command, identically to training's post-action update.
        self.alpha=float(np.clip(self.alpha+np.clip(desired-self.alpha,-.04,.04),0,1))
        return result


class DirectBodySinglePolicy(SinglePolicy):
    def goal(self,features):
        base=features[:,:1323]
        neutral=super().goal(base)
        state,priv=features[:,1323:1375],features[:,1375:1748]
        delta=(4*base[:,102:105]-base[:,52:55])*base.new_tensor([1.,1.,0.])
        delta=delta*(.6/delta.norm(dim=-1).clamp_min(1e-8)).clamp(max=1)[:,None]
        # Replace reference-heading root velocity with measured-heading
        # desired velocity, then add bounded positional/velocity feedback.
        local_delta=base[:,111:114]+delta-priv[:,223:226]
        yaw=torch.atan2(base[:,107],base[:,105])
        omega=torch.stack((torch.zeros_like(yaw),torch.zeros_like(yaw),(4*yaw).clamp(-.8,.8)),-1)
        positions=torch.cat((torch.zeros_like(omega)[:,None],priv[:,1:73].reshape(-1,24,3)),1)
        correction=local_delta[:,None]+torch.cross(omega[:,None].expand_as(positions),positions,dim=-1)
        priv=torch.cat((priv[:,:223],priv[:,223:298]+correction.flatten(1),
            priv[:,298:373]+omega[:,None].expand(-1,25,-1).flatten(1)),-1)
        state=torch.cat((state[:,:-3],state[:,-3:]+omega),-1)
        x=torch.cat((self.normalize('state',state),self.normalize('privileged_state',priv)),-1)
        x=self.linear('_backward_map.net.0',x)
        x=torch.tanh(self.norm('_backward_map.net.1',x))
        direct=16*F.normalize(self.linear('_backward_map.net.3',x),dim=-1)
        adapter=self.goal_adapter((base-self.mean)/self.scale)
        alpha=features[:,1748:1749].clamp(0,1)
        return 16*F.normalize((1-alpha)*neutral+alpha*(direct+adapter),dim=-1)
