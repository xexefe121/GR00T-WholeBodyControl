"""Pure state perturbations and the original committed teacher map. No engine/model."""
import ast
from types import SimpleNamespace
import numpy as np

GROUPS=(('root_position',0,3,'m'),('root_rotation',3,6,'rad'),('joint_position',6,29,'rad'),
        ('root_linear_velocity',29,32,'m/s'),('root_angular_velocity',32,35,'rad/s'),
        ('joint_velocity',35,58,'rad/s'))
KEPT=np.r_[0:52,75:1023]
SIGNS=(-1.,1.)

def core_functions(path):
    tree=ast.parse(path.read_text())
    body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('quat_mul','quat_exp','quat_log')]
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Planner')
    body.append(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='difference'))
    scope={'np':np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(path),'exec'),scope)
    fake=SimpleNamespace(nq=30,nv=29,lin_q=np.r_[0:3,7:30],lin_v=np.r_[0:3,6:29],free=[(0,0)])
    difference=lambda x,y:scope['difference'](fake,x[None],y[None])[0]
    return difference,scope['quat_mul'],scope['quat_exp']

def axis_metadata(native_velocity):
    caps=np.asarray(native_velocity,np.float64)
    if caps.shape!=(23,) or not np.isfinite(caps).all() or not np.all(caps>0):raise ValueError('native speed caps')
    radius=np.r_[np.full(3,.001),np.full(3,.01),np.full(23,.01),np.full(3,.05),np.full(3,.25),.01*caps]
    group=np.concatenate([np.full(b-a,i,np.int8) for i,(_,a,b,_) in enumerate(GROUPS)])
    units=np.concatenate([np.full(b-a,unit,dtype='U5') for _,a,b,unit in GROUPS])
    return radius,group,units

def perturb(qpos,qvel,plan,axis,sign,radius,difference,quat_mul,quat_exp):
    """One signed coordinate in the original plan's tangent chart; no state clamp."""
    if not (0<=axis<58) or sign not in SIGNS or not np.isfinite(radius) or radius<=0:raise ValueError('axis/sign/radius')
    q=np.asarray(qpos,np.float64).copy();v=np.asarray(qvel,np.float64).copy()
    if q.shape!=(30,) or v.shape!=(29,) or np.asarray(plan).shape!=(59,):raise ValueError('state shape')
    if axis<3:q[axis]+=sign*radius
    elif axis<6:
        original=difference(plan,np.r_[q,v])[3:6]
        rotvec=original.copy();rotvec[axis-3]+=sign*radius
        if np.linalg.norm(rotvec)>=np.pi-1e-6:raise ValueError('quaternion principal-chart boundary')
        quat=quat_mul(np.asarray(plan[3:7]),quat_exp(rotvec))
        q[3:7]=quat/np.linalg.norm(quat)
    elif axis<29:q[7+axis-6]+=sign*radius
    else:v[axis-29]+=sign*radius
    return q,v

def committed_target(difference,plan_state,plan_target,gain,qpos,qvel,limits):
    actual=np.r_[qpos,qvel]
    tangent=difference(plan_state,actual)
    raw=gain@tangent
    correction=np.clip(raw,-.1,.1)
    preclip=plan_target+correction
    target=np.clip(preclip,limits[:,0],limits[:,1])
    return target,raw,correction,preclip

def validate_state(q,v,limits,caps):
    if not np.isfinite(q).all() or not np.isfinite(v).all():raise ValueError('nonfinite state')
    if np.any(q[7:]<limits[:,0]) or np.any(q[7:]>limits[:,1]):raise ValueError('strict native joint-position bound')
    if np.any(np.abs(v[6:])>=caps):raise ValueError('strict native joint-speed bound')
    if q[2]<=0:raise ValueError('nonpositive root height')
    if abs(np.linalg.norm(q[3:7])-1)>1e-10:raise ValueError('nonunit quaternion')

def validate_tangent(difference,plan,q0,v0,q,v,axis,signed_radius):
    before=difference(plan,np.r_[q0,v0]);after=difference(plan,np.r_[q,v])
    delta=after-before;expected=np.zeros(58);expected[axis]=signed_radius
    if not np.allclose(delta,expected,rtol=0,atol=2e-12):raise ValueError('not requested single teacher tangent axis')
    return after,delta

def cell_ids(dataset,control):
    dataset=np.asarray(dataset);control=np.asarray(control)
    if not np.array_equal(dataset,np.repeat(np.arange(3),1019)):raise ValueError('dataset order')
    if not np.array_equal(control,np.tile(np.arange(250,1269),3)):raise ValueError('center controls')
    phase=np.where(control<350,0,np.where(control<1169,1,2)).astype(np.int8)
    return phase,(dataset*3+phase).astype(np.int8)
