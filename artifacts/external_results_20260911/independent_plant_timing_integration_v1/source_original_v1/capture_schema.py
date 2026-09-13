"""Fixed little-endian binary64 snapshot. No model/data objects are retained."""
from dataclasses import dataclass
import sys
import numpy as np
from clock_core import CapturedStep,WarningLedger

INTEGRATION_SIZE=291
PACKED_SIZE=373
PACKED_BYTES=PACKED_SIZE*8
STATE_SPEC=8191


def exact(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def typed_vector(value,size,name,finite=False):
    if type(value) is not np.ndarray or value.dtype!=np.float64 or value.shape!=(size,):
        raise ValueError(name+' must be native float64['+str(size)+']')
    if finite and not np.isfinite(value).all():raise ValueError(name+' must be finite')
    return value.copy()


def warning_ledger(counts,lastinfo):
    for name,array in [('counts',counts),('lastinfo',lastinfo)]:
        if type(array) is not np.ndarray or array.dtype!=np.int32 or array.shape!=(8,):
            raise ValueError('native int32 warning '+name+'[8] required')
    if np.any(counts<0):raise ValueError('negative warning count')
    return WarningLedger(tuple(int(v) for v in counts),tuple(int(v) for v in lastinfo))


def pack(integration,qpos,qvel,actuator_force):
    if sys.byteorder!='little':raise ValueError('selected snapshot format is little endian')
    parts=[typed_vector(value,n,name) for value,n,name in
           [(integration,291,'integration'),(qpos,30,'qpos'),(qvel,29,'qvel'),(actuator_force,23,'actuator_force')]]
    if parts[0][1:31].tobytes()!=parts[1].tobytes() or parts[0][31:60].tobytes()!=parts[2].tobytes():
        raise ValueError('capture integration and measured state disagree')
    return np.concatenate(parts).tobytes()


@dataclass(frozen=True)
class Decoded:
    integration: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    actuator_force: np.ndarray


def decode(payload):
    if type(payload) is not bytes or len(payload)!=PACKED_BYTES:
        raise ValueError('owned packed native capture must have exactly2984 bytes')
    values=np.frombuffer(payload,dtype='<f8')  # Read-only because backing bytes are immutable.
    return Decoded(values[:291],values[291:321],values[321:350],values[350:373])


@dataclass(frozen=True)
class Assessment:
    reasons: tuple
    physics_step: int
    joint_excess_rad: float
    speed_ratio: float
    effort_ratio: float
    tilt_rad: float
    clock_error_seconds: float
    ideal_clock_difference_seconds: float
    worst_joint_index: int

    @property
    def issue(self):return '|'.join(self.reasons) if self.reasons else None


def assess_capture(capture,limits,velocity,effort,expected_time,start_time,step):
    """Original0027210c checks, calculated only from immutable captured values."""
    if type(step) is not int or step<0:raise ValueError('nonnegative integer physics step')
    if type(expected_time) is not float or not np.isfinite(expected_time) or expected_time<0:
        raise ValueError('finite nonnegative independently accumulated clock')
    if type(start_time) is not float or not np.isfinite(start_time) or start_time<0:
        raise ValueError('finite nonnegative initial clock')
    values=decode(capture.state);q,dq=values.qpos,values.qvel
    finite=bool(np.isfinite(q).all() and np.isfinite(dq).all())
    if finite:
        excess_by_joint=np.maximum(np.maximum(limits[:,0]-q[7:],q[7:]-limits[:,1]),0.)
        excess=float(excess_by_joint.max())
        speed=float(np.max(np.abs(dq[6:])/velocity))
        tilt=float(np.arccos(np.clip(1-2*(q[4]**2+q[5]**2),-1,1)))
    else:
        excess,speed,tilt=float('inf'),float('inf'),float('inf')
        excess_by_joint=np.full(23,np.inf)
    force=values.actuator_force
    force_finite=bool(force.shape==(23,) and np.isfinite(force).all())
    effort_ratio=float(np.max(np.abs(force)/effort)) if force_finite and step else 0.
    clock_error=abs(float(capture.simulation_time)-expected_time)
    ideal_clock_difference=abs(float(capture.simulation_time)-(start_time+step*.002))
    reasons=[]
    if not finite:reasons.append('nonfinite_state')
    if finite and abs(float(np.linalg.norm(q[3:7]))-1.)>1e-10:reasons.append('invalid_root_quaternion')
    if excess>1e-6:reasons.append('native_joint_bound')
    if speed>1.:reasons.append('native_joint_speed')
    if step and (not force_finite or effort_ratio>1+1e-9):reasons.append('native_actuator_effort')
    if q[2]<.25 or tilt>1.2:reasons.append('fall')
    if np.any(capture.warnings.counts):reasons.append('engine_warning')
    if not np.isfinite(capture.simulation_time) or clock_error>1e-10:reasons.append('physics_clock')
    return Assessment(tuple(reasons),step,excess,speed,effort_ratio,tilt,clock_error,
                      ideal_clock_difference,int(np.argmax(excess_by_joint)))
