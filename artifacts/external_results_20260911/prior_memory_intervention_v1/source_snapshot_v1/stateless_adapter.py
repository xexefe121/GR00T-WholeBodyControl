"""Execute only frozen pure BFM/goal expressions and three pinned ONNX graphs."""
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence
import numpy as np
from scipy.spatial.transform import Rotation

FROZEN=Path(__file__).parent/'frozen'
def definitions(path,names,scope,method_class=None):
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    if method_class:
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==method_class)
        nodes.extend(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names)
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),scope)
    return scope

def prepare(original,motion,original29,contract,sessions):
    observed=definitions(FROZEN/'g1_true23_bfm_seed_observations.py',
        ['_quaternion_matrix','state_and_terms','reference_features'],dict(np=np,Sequence=Sequence))
    source_goal=definitions(FROZEN/'g1_true23_mjbatch_bfm_seed.py',['_goal'],
        dict(np=np,_quaternion_matrix=observed['_quaternion_matrix']),method_class='Native23BFMRolloutSeed')
    task=definitions(FROZEN/'g1_true23_mpc_student.py',['GoalFeatures'],
        dict(np=np,Rotation=Rotation,OFFSETS=np.asarray([0,1,2,4,8,16,24,37]),FEATURES=1023))
    linear=definitions(FROZEN/'student_linear_runtime.py',['LinearFeatures'],
        dict(np=np,GoalFeatures=task['GoalFeatures'],FEATURES=1069))
    c={key:np.asarray(contract[key],dtype=np.float64).copy() for key in ('default_q','kp','kd','training_effort','native_effort','native_velocity')}
    c['body_names']=tuple(contract['body_names'])
    original={k:np.asarray(v).copy() for k,v in original.items()}
    state,privileged=observed['reference_features'](original,c)
    for value in original.values():value.setflags(write=False)
    state.setflags(write=False);privileged.setflags(write=False)
    holder=SimpleNamespace(state=state,privileged=privileged,motion=original,sessions=sessions)
    builder=linear['LinearFeatures'](motion,original29,contract)
    def latent(frame,qpos):return source_goal['_goal'](holder,frame,qpos)
    def sensed(qpos,qvel,prior):return observed['state_and_terms'](qpos[7:],qvel[6:],qpos[3:7],qvel[3:6],prior,c['default_q'])[0]
    return c,builder,latent,sensed
