"""Bounded received-only search with factory balance feedback in predictions.

Synchronous prototype: computation is measured and cannot qualify real-time use.
"""
import time,hashlib,ctypes as ct
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from gear_sonic.utils.g1_true23_feedback_shoot import NativeFeedbackLookahead,F,fptr
from gear_sonic.utils.g1_true23_native_shoot import D,ptr

REFERENCE_FIELDS=('joint','joint_velocity','root','root_rotation','root_velocity','root_omega',
    'feet','feet_velocity','tasks','task_rotation','task_velocity','task_omega')

def predicted_received_goals(reference,controls):
    """Eight-slot input history, using owned packets and declared extrapolation."""
    count=len(reference['joint']);times=np.arange(1,controls+1)*.02
    future={k:np.repeat(np.asarray(reference[k])[-1:],controls,axis=0) for k in REFERENCE_FIELDS}
    for position,velocity in (('joint','joint_velocity'),('root','root_velocity'),('feet','feet_velocity'),('tasks','task_velocity')):
        shape=(controls,)+tuple(1 for _ in future[position].shape[1:])
        future[position]+=times.reshape(shape)*np.asarray(reference[velocity])[-1]
    future['root_rotation']=(Rotation.from_rotvec(times[:,None]*reference['root_omega'][-1])*
        Rotation.from_matrix(reference['root_rotation'][-1])).as_matrix()
    angular=(times[:,None,None]*reference['task_omega'][-1]).reshape(-1,3)
    base=np.repeat(np.asarray(reference['task_rotation'])[-1:],[controls],axis=0).reshape(-1,3,3)
    future['task_rotation']=(Rotation.from_rotvec(angular)*Rotation.from_matrix(base)).as_matrix().reshape(controls,3,3,3)
    slots=np.maximum(0,count-1+np.arange(controls+1)[:,None]+np.array([0,-1,-2,-4,-8,-16,-24,-37]))
    packed=np.concatenate([np.concatenate((reference[k],future[k]),axis=0)[slots].reshape(controls+1,8,-1) for k in REFERENCE_FIELDS],axis=-1)
    if packed.shape!=(controls+1,8,130) or not np.isfinite(packed).all():raise ValueError('Invalid received prediction history')
    return np.ascontiguousarray(packed,np.float64)

class NativeFeedbackPlanner(NativeFeedbackLookahead):
    def __init__(self,*args,learned_checkpoint=None,actor=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.candidates=np.zeros((32,23));self.scores=np.zeros(32);self.terminals=np.zeros((32,59))
        self.rng=np.random.default_rng(20260913)
        self.correction=np.zeros(12);self.applied=np.zeros(12)
        self.last_plan_time=None;self.plans=0;self.plan_interval=.1
        self.learned_checkpoint=None
        if learned_checkpoint:
            import torch
            checkpoint=Path(learned_checkpoint)
            if actor is None or hashlib.sha256(checkpoint.with_suffix('.onnx').read_bytes()).digest()!=hashlib.sha256(Path(actor).read_bytes()).digest():
                raise ValueError('Planner checkpoint and live ONNX export must match')
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False);state=saved['actor']
            if saved['request'].get('neutral_recovery_gate',False):raise ValueError('Native prediction requires the ungated retained controller')
            names=[f'backbone.memory.{i}' for i in (0,2,4)]+[f'backbone.actor.{i}' for i in (0,2,4,6)]
            backbone=np.concatenate([state[n+'.'+kind].numpy().ravel() for n in names for kind in ('weight','bias')])
            np.testing.assert_array_equal(backbone,self.factory_weights)
            np.testing.assert_array_equal(state['default'].numpy(),self.policy.default)
            np.testing.assert_array_equal(state['limits'].numpy(),np.asarray(self.limits,np.float32))
            weights=np.ascontiguousarray(np.concatenate([state[f'goal_head.{i}.{kind}'].numpy().ravel() for i in (0,2,4) for kind in ('weight','bias')]),np.float32)
            mean=np.ascontiguousarray(state['goal_mean'].numpy());scale=np.ascontiguousarray(state['goal_scale'].numpy())
            defaults=np.ascontiguousarray(self.contract['default_q'],np.float64)
            self.lib.feedback_set_learned.argtypes=[ct.c_void_p,F,ct.c_int,F,F,D];self.lib.feedback_set_learned.restype=ct.c_int
            self.lib.feedback_set_received_goals.argtypes=[ct.c_void_p,D,ct.c_int];self.lib.feedback_set_received_goals.restype=ct.c_int
            self.lib.feedback_learned_raw.argtypes=[ct.c_void_p,F,F,F,F];self.lib.feedback_learned_raw.restype=ct.c_int
            self.lib.feedback_received_features.argtypes=[ct.c_void_p,D,D,D,F];self.lib.feedback_received_features.restype=ct.c_int
            if self.lib.feedback_set_learned(self.context,fptr(weights),len(weights),fptr(mean),fptr(scale),ptr(defaults)):
                raise RuntimeError('Native learned-head initialization failed')
            self.learned_checkpoint=str(checkpoint)

    def refine(self,qpos,qvel,baseline,reference,now):
        start=time.perf_counter_ns()
        if self.last_plan_time is None or now-self.last_plan_time>=self.plan_interval-1e-9:
            r={k:v[-1] for k,v in reference.items()}
            if self.learned_checkpoint:
                self.received_goals=predicted_received_goals(reference,self.steps//10)
                if self.lib.feedback_set_received_goals(self.context,ptr(self.received_goals),len(self.received_goals)):
                    raise RuntimeError('Native received prediction upload failed')
            quat=Rotation.from_matrix(r['root_rotation']).as_quat()[[3,0,1,2]]
            goal=np.ascontiguousarray(np.r_[r['joint'],r['joint_velocity'],r['root'],r['root_velocity'],quat,r['root_omega'],
                r['feet'].ravel(),r['feet_velocity'].ravel(),r['tasks'].ravel(),r['task_velocity'].ravel(),
                r['task_rotation'].ravel(),r['task_omega'].ravel()],np.float64)
            q=np.ascontiguousarray(qpos,np.float64);v=np.ascontiguousarray(qvel,np.float64)
            history=np.ascontiguousarray(self.policy.history,np.float32);command=np.ascontiguousarray(self.policy.last_command,np.float32)
            gait=np.ascontiguousarray(np.r_[self.policy.phase,float(self.policy.walking),command[5:]/.2],np.float64)
            lower=self.limits[:12,0]+.06-baseline[:12];upper=self.limits[:12,1]-.06-baseline[:12]
            center=np.clip(self.correction,lower,upper)
            scale=(self.limits[:12,1]-self.limits[:12,0])*.06
            goal_delta=r['joint'][:12]+self.kd[:12]/self.kp[:12]*r['joint_velocity'][:12]-baseline[:12]
            best=np.zeros(12);best_cost=np.inf;baseline_cost=None;feasible=0
            for generation in range(2):
                offsets=self.rng.normal(center,scale,size=(32,12))
                offsets[0]=0.;offsets[1]=center;offsets[2]=best
                offsets[3]=goal_delta;offsets[4]=goal_delta*.5
                offsets=np.clip(offsets,lower,upper)
                self.candidates[:]=baseline;self.candidates[:,:12]+=offsets
                selected=self.lib.feedback_evaluate(self.context,ptr(q),ptr(v),ptr(self.candidates),32,self.steps,ptr(goal),
                    fptr(history),fptr(command),ptr(gait),ptr(self.scores),ptr(self.terminals))
                if selected<0 or not np.isfinite(self.scores).all():raise RuntimeError('Invalid feedback search result')
                if baseline_cost is None:baseline_cost=float(self.scores[0])
                valid=np.flatnonzero(self.scores<1e9);feasible+=len(valid)
                if not len(valid):continue
                order=valid[np.argsort(self.scores[valid])]
                if self.scores[order[0]]<best_cost:
                    best_cost=float(self.scores[order[0]]);best=offsets[order[0]].copy()
                elite=offsets[order[:6]]
                center=elite.mean(axis=0)
                scale=np.maximum(elite.std(axis=0),(self.limits[:12,1]-self.limits[:12,0])*.005)
            self.correction=best if np.isfinite(best_cost) else np.zeros(12)
            self.last_plan_time=float(now);self.plans+=1
            self.status=dict(plans=self.plans,prediction_steps=self.steps,prediction_seconds=self.steps*.002,
                prediction_worlds=32,generations=2,feasible_queries=feasible,
                predicted_cost=best_cost if np.isfinite(best_cost) else None,baseline_predicted_cost=baseline_cost,
                plan_ms=(time.perf_counter_ns()-start)*1e-6,planning_interval_seconds=self.plan_interval,
                predicted_policy=('native12 factory, full retained learned head and actual-target filter at50Hz; fixed candidate offset'
                    if self.learned_checkpoint else 'native12 factory feedback plus fixed current learned-head/candidate offset'),
                learned_checkpoint=self.learned_checkpoint,
                future_reference_frames=0,reference_prediction='constant velocity from received backward derivatives',
                actual_physics_modified=False,synchronous_diagnostic=True)
        # Apply the scored first target. Adding an unmodelled slew here would
        # execute a different first command from every candidate prediction.
        self.applied[:]=self.correction
        target=np.array(baseline,copy=True);target[:12]+=self.applied
        target=np.clip(target,self.limits[:,0]+.06,self.limits[:,1]-.06)
        self.status.update(plan_age_seconds=float(now-self.last_plan_time),
            correction_max_rad=float(np.max(np.abs(self.applied))))
        return target

    def clear_correction(self):
        self.correction[:]=0.;self.applied[:]=0.;self.last_plan_time=None
