"""Offline range-complete student around the frozen native23 ONNX BFM prior."""
import hashlib
import json
from pathlib import Path
import sys
from types import MethodType
import time

import numpy as np
from gear_sonic.utils.g1_true23_mjbatch_bfm_seed import Native23BFMRolloutSeed
from gear_sonic.utils.g1_true23_mpc_student import GoalFeatures, OFFSETS
from terminal_yaw4_goal import terminal_goal_yaw4

BASE=Path(__file__).resolve().parent.parent
if sys.platform=='win32':
    ROOT=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
    TASK=Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910')
    DEPS=None
else:
    ROOT=Path('/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
    TASK=Path('/mnt/e/codex-artifacts/sonic23_teleop_six_hour_20260910')
    DEPS=Path('/mnt/e/codex_sonic_runtime/bfm_seed_20260910/onnx_deps')
BUNDLE=ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'
ONNX=ROOT/'artifacts/teleop_six_hour_20260910/bfm_onnx_v2'
REFERENCE=TASK/'mjbatch_intent_floor_inputs_v1/walk003/reference.npz'
TEACHER=TASK/'bfm_online_intent_v2/walk003_terminal_bfm_yaw4_hybrid_v1'
FEATURES=1069
KIND='native23_bfm_linear_span_student_nominal_v2'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archive(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}


def assert_frozen():
    receipt=json.loads((BASE/'frozen_inputs.json').read_text())
    for name,expected in receipt['source_sha256'].items():
        if sha(BASE/'source_snapshot'/name)!=expected:raise ValueError('frozen pilot source changed: '+name)
    for name,expected in receipt['input_sha256'].items():
        local=name
        if sys.platform!='win32' and len(name)>2 and name[1]==':':
            local='/mnt/'+name[0].lower()+name[2:]
        if sha(local)!=expected:raise ValueError('frozen pilot input changed: '+name)
    return receipt


class LinearFeatures:
    def __init__(self,motion,original29,c):
        self.goals=GoalFeatures(motion,original29,c)
        self.c=c
        self.default=np.asarray(c['default_q'])
        self.limits=np.asarray(c['joint_limits'])

    def __call__(self,qpos,qvel,frame,base_target,previous_action):
        c=self.c
        previous_target=np.clip(self.default+previous_action*.25*np.asarray(c['training_effort'])/np.asarray(c['kp']),self.limits[:,0],self.limits[:,1])
        x=np.r_[self.goals(qpos,qvel,previous_target,frame),base_target-self.default,previous_action].astype(np.float32)
        if x.shape!=(FEATURES,) or not np.isfinite(x).all():raise ValueError('invalid linear student features')
        return x


def infer_base(seed,qpos,qvel,previous_action,history,frame,terminal=False):
    sensed,_=seed._terms(qpos,qvel,previous_action)
    goal=terminal_goal_yaw4(seed,frame,qpos) if terminal else seed._goal(frame,qpos)
    raw=seed.sessions['actor'].run(None,dict(state=sensed[None],last_action=previous_action[None],history=history[None],z=goal))[0][0]*5
    c=seed.contract
    target=c['default_q']+raw*.25*c['training_effort']/c['kp']
    return raw,target,sensed


class LinearStudentRuntime:
    def __init__(self,native,c,original,motion,original29,head_path):
        self.seed=Native23BFMRolloutSeed(native,c,original,ONNX,dependency_directory=DEPS,threads=1)
        self.features=LinearFeatures(motion,original29,c)
        self.c=c;self.limits=np.asarray(c['joint_limits'])
        import onnxruntime as ort
        settings=ort.SessionOptions();settings.intra_op_num_threads=1;settings.inter_op_num_threads=1
        settings.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        self.head=ort.InferenceSession(str(head_path),sess_options=settings,providers=['CPUExecutionProvider'])
        self.head_path=Path(head_path)

    def propose(self,control,qpos,qvel,terminal=False,disable_head=False):
        tick=time.perf_counter()
        if control!=self.seed.recorded_controls:raise ValueError('student control clock mismatch')
        previous=self.seed.previous_action.copy()
        _,terms=self.seed._terms(qpos,qvel,previous)
        history=self.seed.history.before_update(terms)
        raw,base,sensed=infer_base(self.seed,qpos,qvel,previous,history,control+11,terminal)
        if terminal:
            x=np.zeros(FEATURES,np.float32);delta=np.zeros(23,np.float32)
        else:
            x=self.features(qpos,qvel,control+11,base,previous)
            delta=np.zeros(23,np.float32) if disable_head else self.head.run(None,{'features':x[None]})[0][0]
        if delta.shape!=(23,) or not np.isfinite(delta).all():raise ValueError('nonfinite or invalid student delta')
        target=np.clip(base+delta,self.limits[:,0],self.limits[:,1])
        combined=(raw+delta*np.asarray(self.c['kp'])/(.25*np.asarray(self.c['training_effort']))).astype(np.float32)
        self.seed.previous_action=combined.copy();self.seed.recorded_controls+=1
        return dict(target=target,base_target=base,delta=delta,previous_action=previous,action=combined,
            state=sensed,history=history,features=x,inference_ms=(time.perf_counter()-tick)*1000)
