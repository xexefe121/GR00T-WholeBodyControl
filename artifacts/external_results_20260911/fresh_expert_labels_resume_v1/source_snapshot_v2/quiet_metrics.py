"""Exact frozen quiet metric functions from qualified yaw4 hybrid."""
import numpy as np

def stat(x):return dict(p50=float(np.percentile(x,50)),p95=float(np.percentile(x,95)),maximum=float(np.max(x)))

QUIET_THRESHOLDS=dict(window_seconds=3,root_XY_error_p95_m=.05,original_heading_abs_p95_deg=5.,
    root_linear_speed_p95_mps=.05,max_joint_speed_p95_radps=.5,max_joint_speed_max_radps=2.,tilt_max_rad=.15)

def yaw(q):
    w,x,y,z=np.asarray(q).T
    return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))

def standing_windows(arrays,motion,original,original29):
    result={}
    for seconds in (1,3):
        steps=min(seconds*500,len(arrays['physics_torque']))
        q,dq=arrays['physics_qpos'][-steps:],arrays['physics_qvel'][-steps:]
        frames=np.repeat(arrays['source_frame'],arrays['physics_substeps'])[-steps:]
        metric=motion['body_pos_w'][frames,0]
        goal=original['body_pos_w'][frames,0]
        wanted29=original29['source_qpos29'][frames]
        heading=yaw(q[:,3:7])-yaw(wanted29[:,3:7])
        heading=np.abs(np.rad2deg(np.arctan2(np.sin(heading),np.cos(heading))))
        tilt=np.arccos(np.clip(1-2*np.sum(q[:,4:6]**2,axis=1),-1,1))
        result[f'last_{seconds}s']=dict(physics_samples=steps,
            interval_seconds=[float(arrays['physics_time'][-steps-1]),float(arrays['physics_time'][-1])],
            declared_v4_root_error_m=stat(np.linalg.norm(q[:,:3]-metric,axis=1)),
            original_BFM_root_error_m=stat(np.linalg.norm(q[:,:3]-goal,axis=1)),
            horizontal_root_error_m=stat(np.linalg.norm(q[:,:2]-wanted29[:,:2],axis=1)),
            original_heading_absolute_error_deg=stat(heading),root_tilt_rad=stat(tilt),
            horizontal_displacement_from_window_start_m=float(np.linalg.norm(q[-1,:2]-q[0,:2])),
            maximum_absolute_joint_speed_radps=stat(np.max(np.abs(dq[:,6:]),axis=1)),
            joint_speed_rms_radps=float(np.sqrt(np.mean(dq[:,6:]**2))),
            root_linear_speed_mps=stat(np.linalg.norm(dq[:,:3],axis=1)),
            final_maximum_joint_speed_radps=float(np.abs(dq[-1,6:]).max()),
            final_declared_v4_root_error_m=float(np.linalg.norm(q[-1,:3]-metric[-1])),
            final_original_BFM_root_error_m=float(np.linalg.norm(q[-1,:3]-goal[-1])))
    return result

def quiet_diagnostic(windows,strict):
    tail=windows['last_3s']
    gates=dict(all_prior_strict_physical_and_warning_gates=bool(strict),
        full_three_second_window=tail['physics_samples']==1500,
        original_root_XY=tail['horizontal_root_error_m']['p95']<=QUIET_THRESHOLDS['root_XY_error_p95_m'],
        original_heading=tail['original_heading_absolute_error_deg']['p95']<=QUIET_THRESHOLDS['original_heading_abs_p95_deg'],
        root_linear_speed=tail['root_linear_speed_mps']['p95']<=QUIET_THRESHOLDS['root_linear_speed_p95_mps'],
        joint_speed_p95=tail['maximum_absolute_joint_speed_radps']['p95']<=QUIET_THRESHOLDS['max_joint_speed_p95_radps'],
        joint_speed_maximum=tail['maximum_absolute_joint_speed_radps']['maximum']<=QUIET_THRESHOLDS['max_joint_speed_max_radps'],
        maximum_tilt=tail['root_tilt_rad']['maximum']<=QUIET_THRESHOLDS['tilt_max_rad'])
    return dict(quiet_standing_diagnostic_pass=all(gates.values()),gates=gates,thresholds=QUIET_THRESHOLDS,
        raw_last_three_second_distributions=tail,existing_fifteen_source_gates_changed=False,hardware_certification=False)
