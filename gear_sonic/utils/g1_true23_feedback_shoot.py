"""Current/past-pose prediction with the native leg balance feedback retained."""
import ctypes as ct
from pathlib import Path
import time
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from gear_sonic.utils.g1_true23_native_shoot import D,ptr

F=ct.POINTER(ct.c_float)
def fptr(x):return x.ctypes.data_as(F)

class NativeFeedbackLookahead:
    def __init__(self,library,model,contract,kp,kd,tasks,steps,*,firmware,policy):
        if mujoco.__version__!='3.2.3':raise ValueError('feedback prediction requires native MuJoCo3.2.3')
        if not 10<=steps<=300 or steps%10:raise ValueError('prediction must contain1..30 complete control periods')
        self.model=model;self.steps=int(steps);self.policy=policy;self.contract=contract
        self.kp=np.asarray(kp);self.kd=np.asarray(kd);self.limits=np.asarray(contract['joint_limits'])
        ids=np.array([model.jnt_bodyid[0],model.body('left_ankle_roll_link').id,model.body('right_ankle_roll_link').id,
            *[model.body(t['target_body']).id for t in tasks]],np.int32)
        arrays=[np.ascontiguousarray(x,np.float64) for x in (kp,kd,contract['native_effort'],contract['native_velocity'],
            self.limits,np.array([t['target_point'] for t in tasks]))]
        names=[f'.mem_encoder.mem_encoder.{i}' for i in (0,2,4)]+[f'.low_level_net.low_level_net.{i}' for i in (0,2,4)]+['act__matmul_converted']
        with np.load(Path(firmware)/'human_loco_trainable_v1/factory_weights.npz',allow_pickle=False) as z:
            weights=np.ascontiguousarray(np.concatenate([z[n+'.'+kind].ravel() for n in names for kind in ('weight','bias')]),np.float32)
        if weights.size!=151276 or not np.isfinite(weights).all():raise ValueError('invalid factory feedback weights')
        self.factory_weights=weights
        defaults=np.ascontiguousarray(policy.default,np.float64);scale=np.ascontiguousarray(policy.scale,np.float64)
        self.lib=ct.CDLL(str(library));self.lib.feedback_create.argtypes=[ct.c_void_p,*[D]*5,ct.POINTER(ct.c_int),D,F,ct.c_int,D,D]
        self.lib.feedback_create.restype=ct.c_void_p;self.lib.feedback_destroy.argtypes=[ct.c_void_p]
        self.lib.feedback_evaluate.argtypes=[ct.c_void_p,D,D,D,ct.c_int,ct.c_int,D,F,F,D,D,D]
        self.lib.feedback_evaluate.restype=ct.c_int
        self.lib.feedback_raw.argtypes=[ct.c_void_p,F,F,F]
        self.context=self.lib.feedback_create(model._address,*[ptr(x) for x in arrays[:5]],ids.ctypes.data_as(ct.POINTER(ct.c_int)),ptr(arrays[5]),
            fptr(weights),len(weights),ptr(defaults),ptr(scale))
        if not self.context:raise ValueError('native feedback prediction initialization failed')
        self.candidates=np.zeros((8,23));self.scores=np.zeros(8);self.terminals=np.zeros((8,59));self.status={}

    def refine(self,qpos,qvel,baseline,reference):
        start=time.perf_counter_ns();r={k:v[-1] for k,v in reference.items()}
        quat=Rotation.from_matrix(r['root_rotation']).as_quat()[[3,0,1,2]]
        goal=np.ascontiguousarray(np.r_[r['joint'],r['joint_velocity'],r['root'],r['root_velocity'],quat,r['root_omega'],
            r['feet'].ravel(),r['feet_velocity'].ravel(),r['tasks'].ravel(),r['task_velocity'].ravel(),
            r['task_rotation'].ravel(),r['task_omega'].ravel()],np.float64)
        if goal.shape!=(125,) or not np.isfinite(goal).all():raise ValueError('invalid received prediction goal')
        self.candidates[:]=baseline
        for row,(blend,lead) in enumerate(((0,0),(.25,0),(.5,0),(.75,0),(1.,0),(.5,.04),(.75,.04),(1.,.04))):
            desired=r['joint'][:12]+(self.kd[:12]/self.kp[:12]+lead)*r['joint_velocity'][:12]
            self.candidates[row,:12]=(1-blend)*baseline[:12]+blend*desired
        np.clip(self.candidates,self.limits[:,0]+.06,self.limits[:,1]-.06,out=self.candidates)
        q=np.ascontiguousarray(qpos,np.float64);v=np.ascontiguousarray(qvel,np.float64)
        history=np.ascontiguousarray(self.policy.history,np.float32);command=np.ascontiguousarray(self.policy.last_command,np.float32)
        gait=np.ascontiguousarray(np.r_[self.policy.phase,float(self.policy.walking),command[5:]/.2],np.float64)
        selected=self.lib.feedback_evaluate(self.context,ptr(q),ptr(v),ptr(self.candidates),8,self.steps,ptr(goal),
            fptr(history),fptr(command),ptr(gait),ptr(self.scores),ptr(self.terminals))
        if selected<0 or not np.isfinite(self.scores).all():raise ValueError('invalid feedback prediction result')
        all_failed=bool(np.min(self.scores)>=1e9)
        if all_failed:selected=0
        self.status=dict(selected_candidate=int(selected),prediction_steps=self.steps,prediction_seconds=self.steps*.002,
            predicted_cost=float(self.scores[selected]),baseline_predicted_cost=float(self.scores[0]),
            all_candidates_predicted_failure=all_failed,lookahead_ms=(time.perf_counter_ns()-start)*1e-6,
            actual_future_reference_frames=0,reference_prediction='received backward velocity extrapolation',
            prediction_feedback='exact native12 factory balance at50Hz; fixed candidate leg offset',
            prediction_worlds=8,actual_physics_modified=False)
        return self.candidates[selected].copy()

    def close(self):
        if self.context:self.lib.feedback_destroy(self.context);self.context=None
