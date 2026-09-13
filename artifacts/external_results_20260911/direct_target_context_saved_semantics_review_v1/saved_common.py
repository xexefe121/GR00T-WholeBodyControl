"""Saved-array operations only; no task runtime, graph or native imports."""
import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence
import numpy as np
from scipy.spatial.transform import Rotation

BASE=Path(__file__).resolve().parent
NEW=BASE.parent
RUN=NEW/'direct_target_causal_context_evaluation_v2'
SOURCE=RUN/'source_draft_v1'
SUBJECTS=('fit_report','checkpoint','head','normalization','training_manifest','training_request','export_manifest','coefficient',
          'source_checkpoint','full_state_generation_request','full_state_generation_report','full_state_data_audit','full_state_data_owner',
          'paired_report','shared_manifest','context_alignment','blinded_fit_report','causal_fit_report')
PARITIES=('canonical_initial_full291_parity','canonical_prefix250_parity','actual_query250_input_parity','actual_query250_ownexport_output_parity')

def local(value):
    value=str(value).replace('\\','/')
    if sys.platform!='win32' and len(value)>2 and value[1]==':':value='/mnt/'+value[0].lower()+value[2:]
    return Path(value)

def key(value):
    value=str(value).replace('\\','/')
    if value.startswith('/mnt/') and value[6:7]=='/':value=value[5]+':'+value[6:]
    return value.casefold()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()

def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def load(path):
    with np.load(path,allow_pickle=False) as a:return {name:a[name].copy() for name in a.files}

def contains(mapping,path,digest):
    converted={}
    for name,value in mapping.items():
        identity=key(name)
        assert identity not in converted or converted[identity]==value,'Conflicting normalized path'
        converted[identity]=value
    assert converted.get(key(path))==digest,str(path)

def pure(paths):
    scope=dict(np=np,Rotation=Rotation,Sequence=Sequence,OFFSETS=np.array([0,1,2,4,8,16,24,37],np.int64),FEATURES=1000)
    for name,names in [('feature_source',{'DirectFeatures'}),('observation_source',{'state_and_terms','_quaternion_matrix'})]:
        tree=ast.parse(paths[name].read_text())
        body=[v for v in tree.body if isinstance(v,(ast.ClassDef,ast.FunctionDef)) and v.name in names]
        assert {v.name for v in body}==names
        exec(compile(ast.Module(body=body,type_ignores=[]),str(paths[name]),'exec'),scope)
    return scope['DirectFeatures'],scope['state_and_terms']

def phase(control):return 0 if control<250 else 1 if control<1269 else 2
def history_zero():
    return {k:np.zeros((4,n),np.float32) for k,n in dict(actions=23,base_ang_vel=3,dof_pos=23,dof_vel=23,projected_gravity=3).items()}
def flat(history):return np.concatenate([history[k].reshape(-1) for k in sorted(history)]).copy()
def advance(history,terms):
    for name in history:history[name][1:]=history[name][:-1].copy();history[name][0]=terms[name]
def actual_action(target,default,kp,effort):return ((target-default)*kp/(.25*effort)).astype(np.float32)
def times(start,steps):
    values=[float(start)]
    for _ in range(steps):values.append(values[-1]+.002)
    return np.asarray(values,np.float64)

class Checks:
    def __init__(self):self.rows=[];self.active=None
    def require(self,condition,name):
        self.active=name;ok=bool(condition);self.rows.append(dict(name=name,passed=ok))
        if not ok:raise AssertionError(name)
    def exact(self,actual,expected,name):
        a,b=np.asarray(actual),np.asarray(expected)
        self.require(a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),name)
