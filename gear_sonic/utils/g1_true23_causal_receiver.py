"""Received-only adapter reusing the established native23 packet fault gate."""
from collections import deque
import numpy as np
from gear_sonic.utils.g1_true23_received_features import prepare_reference, features_numpy, MeasuredHistory


class CausalReceiver:
    def __init__(self, contract, *, now=0., model=None, standing_qpos=None, tasks=None):
        from gear_sonic.utils.g1_true23_bfmzero_stream import PacketGate
        self.gate=PacketGate(contract,now=now)
        self.contract=contract
        self.samples=deque(maxlen=39) # one extra frame for oldest backward derivative
        self.tasks=deque(maxlen=39)
        self.reference_terms=deque(maxlen=39)
        self.cached_reference=None
        self.history=MeasuredHistory(contract)
        self.model,self.standing_qpos,self.task_spec=model,standing_qpos,tasks
        self.stop=None;self.mode='source';self.stop_last_time=None

    def receive(self, packet, task_position, task_quaternion, now):
        position,quaternion=np.asarray(task_position),np.asarray(task_quaternion)
        if position.shape!=(3,3) or quaternion.shape!=(3,4) or not np.isfinite(position).all() or not np.isfinite(quaternion).all():
            self.gate.latch('invalid_hand_head_reference',now);return False
        if np.max(np.abs(np.linalg.norm(quaternion,axis=-1)-1))>1e-5:
            self.gate.latch('invalid_task_quaternion',now);return False
        if not self.gate.receive(packet,now):return False
        if self.gate.anchor is not None:
            from scipy.spatial.transform import Rotation
            anchor=self.gate.anchor
            transformed=anchor.rotation.apply(position-anchor.original_origin)+anchor.measured_origin
            transformed[:,2]=position[:,2]
            quaternion=(anchor.rotation*Rotation.from_quat(quaternion[:,[1,2,3,0]])).as_quat()[:,[3,0,1,2]]
            position=transformed
        # Unlike the BFM H8 preview reader, this actor needs just the newly
        # received sample and its owned past. No future-window readiness wait.
        sample=self.gate.consume()
        self.append_owned(sample.packet.fields,position.copy(),quaternion.copy())
        return True

    def append_owned(self,fields,position,quaternion):
        """Compute received derivatives once at admission from two owned poses."""
        previous=[self.samples[-1]] if self.samples else []
        pairs=[self.tasks[-1]] if self.tasks else []
        motion={k:np.stack([s[k] for s in [*previous,fields]]) for k in fields}
        pairs.append((position,quaternion))
        original=dict(source_task_position_w=np.stack([t[0] for t in pairs]),
                      source_task_quaternion_wxyz=np.stack([t[1] for t in pairs]))
        reference=prepare_reference(motion,original,self.contract)
        self.reference_terms.append({k:v[-1].copy() for k,v in reference.items()})
        self.samples.append(fields);self.tasks.append((position,quaternion))
        self.cached_reference=None

    def reference(self,now):
        self.gate.check_freshness(now)
        if not self.samples:raise ValueError('reference unavailable')
        if self.gate.fault:
            if self.model is None:raise ValueError('fault latched without standing reference generator')
            if self.stop is None:
                from gear_sonic.utils.g1_true23_causal_stop import StandingReference
                self.stop=StandingReference(self.model,self.samples[-1],self.standing_qpos,self.task_spec)
                self.mode='stopping'
            if self.stop_last_time is None or now-self.stop_last_time>=.02-1e-9:
                fields,pos,quat=self.stop.next()
                self.append_owned(fields,pos,quat);self.stop_last_time=now
            if self.stop.step>=100:self.mode='latched_standing'
        if self.cached_reference is None:
            self.cached_reference={k:np.stack([r[k] for r in self.reference_terms]) for k in self.reference_terms[0]}
            # Preserve the old finite-window convention exactly. The oldest
            # row has no earlier sample in that window; actor history offsets
            # use the next row once the complete 39-sample window is present.
            for key in ('joint_velocity','root_velocity','root_omega','feet_velocity','task_velocity','task_omega'):
                self.cached_reference[key][0]=0
        return self.cached_reference

    def features(self,qpos,qvel,now):
        ref=self.reference(now)
        return features_numpy(qpos,qvel,ref,len(self.samples)-1,self.contract['default_q'],
                              self.history.prior,self.history.vector())

    def commit(self,qpos,qvel,target):self.history.commit(qpos,qvel,target)

    def rearm(self,now,measured_qpos):
        self.gate.rearm(now,measured_qpos)
        self.samples.clear();self.tasks.clear()
        self.reference_terms.clear();self.cached_reference=None
        self.stop=None;self.mode='source';self.stop_last_time=None
        # Robot history stays continuous across input rearm.
