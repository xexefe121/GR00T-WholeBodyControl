"""Received-only short dynamics lookahead around a native23 controller."""
import ctypes as ct
import time
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

D=ct.POINTER(ct.c_double)
def ptr(x):return x.ctypes.data_as(D)

class NativePoseLookahead:
    def __init__(self,library,model,contract,kp,kd,tasks,steps=30):
        if mujoco.__version__!='3.2.3':raise ValueError('lookahead requires native MuJoCo3.2.3')
        self.model=model;self.steps=int(steps)
        self.kp=np.asarray(kp);self.kd=np.asarray(kd);self.limits=np.asarray(contract['joint_limits'])
        ids=np.array([model.jnt_bodyid[0],model.body('left_ankle_roll_link').id,model.body('right_ankle_roll_link').id,
            *[model.body(t['target_body']).id for t in tasks]],np.int32)
        arrays=[np.ascontiguousarray(x,np.float64) for x in (kp,kd,contract['native_effort'],contract['native_velocity'],
            self.limits,np.array([t['target_point'] for t in tasks]))]
        self.lib=ct.CDLL(str(library));self.lib.shoot_create.argtypes=[ct.c_void_p,*[D]*5,ct.POINTER(ct.c_int),D]
        self.lib.shoot_create.restype=ct.c_void_p;self.lib.shoot_destroy.argtypes=[ct.c_void_p]
        self.lib.shoot_evaluate.argtypes=[ct.c_void_p,D,D,D,ct.c_int,ct.c_int,D,D,D]
        self.lib.shoot_evaluate.restype=ct.c_int
        self.context=self.lib.shoot_create(model._address,*[ptr(x) for x in arrays[:5]],ids.ctypes.data_as(ct.POINTER(ct.c_int)),ptr(arrays[5]))
        if not self.context:raise ValueError('native lookahead initialization failed')
        self.candidates=np.zeros((8,23));self.scores=np.zeros(8);self.terminals=np.zeros((8,59));self.status={}

    def refine(self,qpos,qvel,baseline,reference):
        start=time.perf_counter_ns();r={k:v[-1] for k,v in reference.items()}
        quat=Rotation.from_matrix(r['root_rotation']).as_quat()[[3,0,1,2]]
        goal=np.ascontiguousarray(np.r_[r['joint'],r['joint_velocity'],r['root'],r['root_velocity'],quat,r['root_omega'],
            r['feet'].ravel(),r['feet_velocity'].ravel(),r['tasks'].ravel(),r['task_velocity'].ravel(),
            r['task_rotation'].ravel(),r['task_omega'].ravel()],np.float64)
        assert goal.shape==(125,)
        self.candidates[:]=baseline
        # Full-range received joint targets are candidates, never a prepared
        # trajectory. The second group uses 40ms constant-velocity prediction
        # from the last received derivative; no later packets are accessible.
        for row,(blend,lead) in enumerate(((0,0),(.25,0),(.5,0),(.75,0),(1.,0),(.5,.04),(.75,.04),(1.,.04))):
            desired=r['joint'][:12]+(self.kd[:12]/self.kp[:12]+lead)*r['joint_velocity'][:12]
            self.candidates[row,:12]=(1-blend)*baseline[:12]+blend*desired
        np.clip(self.candidates,self.limits[:,0]+.06,self.limits[:,1]-.06,out=self.candidates)
        q=np.ascontiguousarray(qpos,np.float64);v=np.ascontiguousarray(qvel,np.float64)
        selected=self.lib.shoot_evaluate(self.context,ptr(q),ptr(v),ptr(self.candidates),len(self.candidates),self.steps,
            ptr(goal),ptr(self.scores),ptr(self.terminals))
        if selected<0 or not np.isfinite(self.scores).all():raise ValueError('invalid native lookahead result')
        all_failed=bool(np.min(self.scores)>=1e9)
        if all_failed:selected=0
        self.status=dict(selected_candidate=int(selected),prediction_steps=self.steps,prediction_seconds=self.steps*.002,
            predicted_cost=float(self.scores[selected]),baseline_predicted_cost=float(self.scores[0]),
            all_candidates_predicted_failure=all_failed,lookahead_ms=(time.perf_counter_ns()-start)*1e-6,
            actual_future_reference_frames=0,reference_prediction='constant velocity from received backward differences')
        return self.candidates[selected].copy()

    def close(self):
        if self.context:self.lib.shoot_destroy(self.context);self.context=None
