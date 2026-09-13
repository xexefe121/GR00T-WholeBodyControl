"""Pure-array checks and canonical trace schema for one offline hybrid trial."""
import numpy as np

SHAPES = dict(qpos=(30,), qvel=(29,), target=(23,), source_frame=(), global_control=(),
    controller_mode=(), joint_error=(23,), root_error=(3,), state=(52,), history=(300,),
    previous_action=(23,), action=(23,), inference_ms=(), range_excess=(), velocity_ratio=(),
    effort_ratio=(), physics_substeps=(), physics_qpos=(30,), physics_qvel=(29,),
    physics_requested_torque=(23,), physics_torque=(23,), physics_actuator_force=(23,),
    physics_time=(), physics_expected_time=(), physics_warning_number=(8,), physics_warning_lastinfo=(8,))
FLOAT32 = {'state', 'history', 'previous_action', 'action'}
INTEGER = {'source_frame', 'global_control', 'controller_mode', 'physics_substeps'}
WARNINGS = {'physics_warning_number', 'physics_warning_lastinfo'}
QUIET = dict(root_xy_p95_m=.05, original_yaw_p95_deg=5., root_linear_speed_p95_mps=.05,
    joint_speed_p95_radps=.5, joint_speed_max_radps=2., tilt_max_rad=.15)

def exact(actual, expected, label):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
        raise ValueError('bit-exact mismatch: ' + label)

def arrays(trace):
    out = {}
    for key, shape in SHAPES.items():
        dtype = np.float32 if key in FLOAT32 else np.int32 if key in WARNINGS else np.int64 if key in INTEGER else np.float64
        out[key] = np.asarray(trace[key], dtype=dtype).reshape((-1,) + shape)
    controls, steps = len(out['target']), len(out['physics_torque'])
    for key in SHAPES:
        expected = steps + 1 if key in ('physics_qpos','physics_qvel','physics_time','physics_expected_time','physics_warning_number','physics_warning_lastinfo') else steps if key.startswith('physics_') and key != 'physics_substeps' else controls + 1 if key in ('qpos','qvel') else controls
        if len(out[key]) != expected:
            raise ValueError('trace length mismatch: ' + key)
    if int(out['physics_substeps'].sum()) != steps:
        raise ValueError('physics substep sum mismatch')
    return out

def flatten_history(history):
    return np.concatenate([history.data[k].reshape(-1) for k in sorted(history.data)]).copy()

def history_snapshot(history, previous, count):
    return dict(previous_action=previous.copy(), recorded_controls=np.asarray(count, np.int64),
        **{'history_' + key: value.copy() for key, value in history.data.items()})

def first_issue(q, dq, force, warning_number, warning_lastinfo, time, expected_time, limits, speed, effort):
    if not all(np.isfinite(v).all() for v in (q, dq, force, np.asarray(time), np.asarray(expected_time))):
        return 'nonfinite_native_state_or_force'
    if np.any(warning_number) or np.any(warning_lastinfo):
        return 'engine_warning'
    if np.asarray(time, np.float64).tobytes() != np.asarray(expected_time, np.float64).tobytes():
        return 'independent_repeated_clock_mismatch'
    if abs(np.linalg.norm(q[3:7]) - 1.) > 1e-10:
        return 'quaternion_norm'
    tilt = float(np.arccos(np.clip(1 - 2*np.sum(q[4:6]**2), -1, 1)))
    if q[2] < .25 or tilt > 1.2:
        return 'fall'
    if np.maximum(limits[:, 0]-q[7:], q[7:]-limits[:, 1]).max() > 1e-6:
        return 'native_joint_range'
    if np.max(np.abs(dq[6:])/speed) > 1.:
        return 'native_joint_speed'
    if np.max(np.abs(force)/effort) > 1.+1e-9:
        return 'native_actuator_effort'
    return None

def prefix_sample(source, sample, global_step):
    for key, value in sample.items():
        index = global_step - 1 if key in ('physics_requested_torque', 'physics_torque', 'physics_actuator_force') else global_step
        exact(value, source[key][index], key + ' at native step ' + str(global_step))

def quiet_metrics(a, original29):
    if len(a['physics_torque']) < 1500:
        return dict(samples=len(a['physics_torque']), gates={}, pass_all=False)
    q, dq = a['physics_qpos'][-1500:], a['physics_qvel'][-1500:]
    frame = np.repeat(a['source_frame'], a['physics_substeps'])[-1500:]
    goal = original29['source_qpos29'][frame]
    def yaw(v):
        w, x, y, z = v.T
        return np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    dy = yaw(q[:, 3:7]) - yaw(goal[:, 3:7])
    joint = np.abs(dq[:, 6:]).max(axis=1)
    values = dict(root_xy_p95_m=float(np.percentile(np.linalg.norm(q[:, :2]-goal[:, :2], axis=1), 95)),
        original_yaw_p95_deg=float(np.percentile(np.abs(np.rad2deg(np.arctan2(np.sin(dy), np.cos(dy)))), 95)),
        root_linear_speed_p95_mps=float(np.percentile(np.linalg.norm(dq[:, :3], axis=1), 95)),
        joint_speed_p95_radps=float(np.percentile(joint, 95)), joint_speed_max_radps=float(joint.max()),
        tilt_max_rad=float(np.arccos(np.clip(1-2*np.sum(q[:, 4:6]**2, axis=1), -1, 1)).max()))
    gates = {k: bool(np.isfinite(v) and v <= QUIET[k]) for k, v in values.items()}
    return dict(samples=1500, **values, gates=gates, pass_all=all(gates.values()), thresholds=QUIET)
