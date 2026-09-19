"""Current received-body BFM encoding; one actor, no future reference window."""
import numpy as np
import torch
import torch.nn.functional as F
from gear_sonic.utils.g1_true23_single_policy import SinglePolicy
from gear_sonic.utils.g1_true23_bfmzero_inference import reference_features
from gear_sonic.utils.g1_true23_received_features import backward_velocity,backward_omega,rotations

INPUT_SIZE=1749


def task_closure_signature(sample,task_position,task_quaternion,contract):
    """Received geometric intent in reference heading, independent of robot state.

    Upper-arm IK branches are not used to infer standing. All three original
    hand/head task positions and orientations remain in the comparison.
    """
    root=sample['body_pos_w'][0]
    rotation=rotations(np.asarray(sample['body_quat_w'][0]))
    yaw=np.arctan2(rotation[1,0],rotation[0,0]);c,s=np.cos(yaw),np.sin(yaw)
    heading=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    feet=[contract['body_names'].index(side+'_ankle_roll_link') for side in ('left','right')]
    return dict(legs_waist=np.asarray(sample['joint_pos'][:13]).copy(),
        feet=(sample['body_pos_w'][feet]-root)@heading,
        tasks=(np.asarray(task_position)-root)@heading,
        task_rotation=heading.T@rotations(np.asarray(task_quaternion)))


def task_closure_standing(signature,anchor,*,joint_velocity,root_velocity,root_omega,
                          feet_velocity,task_velocity,task_omega):
    """Pure received-only closure predicate shared by runtime and bank building."""
    stationary=(np.max(np.abs(joint_velocity))<=.10 and np.linalg.norm(root_velocity)<=.025
        and np.linalg.norm(root_omega)<=.10 and np.linalg.norm(np.asarray(feet_velocity).reshape(-1,3),axis=-1).max()<=.025
        and np.linalg.norm(np.asarray(task_velocity).reshape(-1,3),axis=-1).max()<=.025
        and np.linalg.norm(np.asarray(task_omega).reshape(-1,3),axis=-1).max()<=.10)
    if not stationary:return False
    if np.max(np.abs(signature['legs_waist']-anchor['legs_waist']))>.04:return False
    if np.linalg.norm(signature['feet']-anchor['feet'],axis=-1).max()>.01:return False
    if np.linalg.norm(signature['tasks']-anchor['tasks'],axis=-1).max()>.01:return False
    relative=signature['task_rotation']@anchor['task_rotation'].transpose(0,2,1)
    angles=np.arccos(np.clip((np.trace(relative,axis1=1,axis2=2)-1)/2,-1,1))
    return bool(np.max(angles)<=.05)


def causal_body_features(motion,contract):
    causal=dict(motion)
    causal['joint_vel']=backward_velocity(motion['joint_pos'])
    causal['body_lin_vel_w']=backward_velocity(motion['body_pos_w'])
    causal['body_ang_vel_w']=backward_omega(motion['body_quat_w'])
    state,priv=reference_features(causal,contract)
    return np.concatenate((state,priv),-1).astype(np.float32)


class ReceivedBodyGoal:
    """Adds current encoded pose and a past-only goal blend to owned samples."""
    def __init__(self,contract,*,task_closure_stand=False):
        self.contract=contract;self.alpha=0.;self.epoch=None;self.first=None
        self.task_closure_stand=bool(task_closure_stand);self.standing_anchor=None

    def features(self,base,receiver):
        alpha=self.advance_blend(base,receiver)
        samples=list(receiver.samples)[-2:]
        motion={k:np.stack([s[k] for s in samples]) for k in samples[-1]}
        encoded=causal_body_features(motion,self.contract)[-1]
        return np.r_[base,encoded,alpha].astype(np.float32)

    def advance_blend(self,base,receiver):
        """Return current blend and advance it once, without a BFM encoding.

        Controllers that consume native body features need only this causal
        geometric state. Building the unused 425-wide BFM pose encoding in
        their critical loop adds cost without changing the command.
        """
        if receiver.gate.epoch!=self.epoch:
            self.epoch=receiver.gate.epoch;self.alpha=0.;self.first=None;self.standing_anchor=None
        if self.first is None:self.first=np.array(receiver.samples[0]['joint_pos'],copy=True)
        if self.task_closure_stand and self.standing_anchor is None:
            self.standing_anchor=task_closure_signature(receiver.samples[0],*receiver.tasks[0],self.contract)
        result=self.alpha
        joint=base[56:79]
        near=min(np.max(np.abs(joint-self.first)),np.max(np.abs(joint-self.contract['default_q'])))<=.04
        moving=(not near) or np.max(np.abs(base[79:102]))>.10 or np.linalg.norm(base[111:114])>.025 or np.linalg.norm(base[114:117])>.10
        if self.task_closure_stand:
            signature=task_closure_signature(receiver.samples[-1],*receiver.tasks[-1],self.contract)
            closed=task_closure_standing(signature,self.standing_anchor,joint_velocity=base[79:102],
                root_velocity=base[111:114],root_omega=base[114:117],feet_velocity=base[123:129],
                task_velocity=base[156:165],task_omega=base[165:174])
            moving=not closed
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
