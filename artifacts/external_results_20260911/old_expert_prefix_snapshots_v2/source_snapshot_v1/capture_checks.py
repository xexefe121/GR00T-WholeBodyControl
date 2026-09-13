"""Pure exact comparisons and explicit old warning serialization."""
import numpy as np

def snapshot_execution_flags(controls,completed,verified,attempted):
    controls=np.asarray(controls,np.int64)
    returned=np.clip(completed-controls*10,0,10)
    passed=np.clip(verified-controls*10,0,10)
    attempts=np.clip(attempted-controls*10,0,10)
    return dict(snapshot_native_steps_completed=returned,snapshot_native_steps_verified=passed,
        snapshot_native_steps_attempted=attempts,snapshot_control_was_started=attempts>0,
        snapshot_control_fully_verified=passed==10)

def exact(a,b,key):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape or a.dtype!=b.dtype or a.tobytes()!=b.tobytes():
        raise ValueError('byte mismatch: '+key)

def original_warning(native,recorded,key):
    native,recorded=np.asarray(native),np.asarray(recorded)
    if native.shape!=(8,) or native.dtype!=np.int32 or recorded.shape!=(8,) or recorded.dtype!=np.int64:
        raise ValueError('Unexpected warning serialization: '+key)
    serialized=native.astype(np.int64)
    exact(serialized,recorded,key)
    return serialized

def compare_sample(sample,trace,step,expected):
    exact(sample['qpos'],trace['physics_qpos'][step],'qpos')
    exact(sample['qvel'],trace['physics_qvel'][step],'qvel')
    original_warning(sample['warning_counts'],trace['physics_warning_counts'][step],'warning counts')
    original_warning(sample['warning_lastinfo'],trace['physics_warning_lastinfo'][step],'warning lastinfo')
    if sample['time']!=trace['physics_time'][step] or sample['time']!=expected:
        raise ValueError('Recorded or independent repeated clock mismatch.')
    if step:
        exact(sample['ctrl'],trace['physics_torque'][step-1],'command torque')
        exact(sample['force'],trace['physics_actuator_torque'][step-1],'actuator force')
