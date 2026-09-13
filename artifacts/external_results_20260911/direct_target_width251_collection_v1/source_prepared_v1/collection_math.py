"""Pure byte checks and qualified committed-map arithmetic, no native imports."""
import ast
from types import SimpleNamespace
import numpy as np

START,STOP,MAIN,HOLD=251,1269,1569,250
WIDTHS={'actions':23,'base_ang_vel':3,'dof_pos':23,'dof_vel':23,'projected_gravity':3}

def exact(a,b,name):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape or a.dtype!=b.dtype or a.tobytes()!=b.tobytes():
        raise ValueError('Saved byte identity: '+name)

def array(a,shape,dtype,name):
    if not isinstance(a,np.ndarray) or a.shape!=shape or a.dtype!=np.dtype(dtype) or not np.isfinite(a).all():
        raise ValueError('Finite exact schema: '+name)
    return a

def difference_function(core):
    tree=ast.parse(core.read_text())
    body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('quat_mul','quat_log')]
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Planner')
    body.append(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='difference'))
    namespace={'np':np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(core),'exec'),namespace)
    holder=SimpleNamespace(nq=30,nv=29,lin_q=np.r_[0:3,7:30],lin_v=np.r_[0:3,6:29],free=[(0,0)])
    return lambda x,y:namespace['difference'](holder,x[None],y[None])[0]

def committed_target(difference,plan_state,plan_target,gain,qpos,qvel,limits):
    # Preserve the reviewed saved_committed_map full product and both nested clips.
    actual=np.r_[qpos,qvel]
    tangent=difference(plan_state,actual)
    raw=gain@tangent
    correction=np.clip(raw,-.1,.1)
    preclip=plan_target+correction
    target=np.clip(preclip,limits[:,0],limits[:,1])
    return target,raw,correction,preclip

def phase(control):
    if not START<=control<STOP:raise ValueError('Only fresh expert controls251..1268')
    return 0 if control<350 else 1 if control<1169 else 2

def plan_index(control):
    phase(control)
    return START+5*((control-START)//5),(control-START)%5

def named(flat):
    array(flat,(300,),'float32','incoming history')
    result={};offset=0
    for key in sorted(WIDTHS):
        size=4*WIDTHS[key];result[key]=flat[offset:offset+size].reshape(4,WIDTHS[key]).copy();offset+=size
    return result
