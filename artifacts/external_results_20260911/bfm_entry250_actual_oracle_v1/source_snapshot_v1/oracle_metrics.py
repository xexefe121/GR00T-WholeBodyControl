import mujoco
import numpy as np
from quiet_metrics import yaw

def get_state(model,data):
    spec=mujoco.mjtState.mjSTATE_INTEGRATION
    value=np.empty(mujoco.mj_stateSize(model,spec))
    mujoco.mj_getState(model,data,value,spec)
    return value

def assess(data,c,expected):
    limits=np.asarray(c['joint_limits']);q,dq=data.qpos,data.qvel
    finite=np.isfinite(q).all() and np.isfinite(dq).all()
    excess=float(np.maximum(0,np.maximum(limits[:,0]-q[7:],q[7:]-limits[:,1])).max()) if finite else float('inf')
    velocity=float(np.max(np.abs(dq[6:])/np.asarray(c['native_velocity']))) if finite else float('inf')
    force=data.qfrc_actuator[6:]
    effort=float(np.max(np.abs(force)/np.asarray(c['native_effort']))) if np.isfinite(force).all() else float('inf')
    tilt=float(np.arccos(np.clip(1-2*np.sum(q[4:6]**2),-1,1))) if finite else float('inf')
    reasons=[]
    if not finite or not np.isfinite(force).all():reasons.append('nonfinite_state_or_force')
    if finite and abs(np.linalg.norm(q[3:7])-1)>1e-10:reasons.append('invalid_root_quaternion')
    if excess>1e-6:reasons.append('native_joint_bound')
    if velocity>1:reasons.append('native_joint_speed')
    if effort>1+1e-9:reasons.append('native_actuator_force')
    if q[2]<.25 or tilt>1.2:reasons.append('fall')
    if np.any(data.warning.number):reasons.append('engine_warning')
    clock=float(abs(data.time-expected))
    if not np.isfinite(data.time) or clock>1e-10:reasons.append('physics_clock')
    return reasons,dict(range_excess=excess,velocity_ratio=velocity,effort_ratio=effort,tilt=tilt,clock_error=clock)

def source_metrics(native,arrays,motion,original29,audit):
    complete=np.asarray(arrays['physics_substeps'])==10
    selected=complete&(arrays['global_control']>=350)&(arrays['global_control']<1169)
    if not np.any(selected):return dict(source_controls=0,no_source_samples=True)
    poses=arrays['qpos'][1:][selected];frames=arrays['source_frame'][selected]
    if not np.isfinite(poses).all():
        return dict(source_controls=int(selected.sum()),unavailable_reason='nonfinite measured source poses; failure retained')
    data=mujoco.MjData(native);points=[];feet=[]
    tasks=[next(t for t in audit['original_intent']['convention']['tasks'] if t['name']==name) for name in ('left_hand','right_hand','head_proxy')]
    ids=[native.body(t['target_body']).id for t in tasks]
    foot_ids=[native.body(side+'_ankle_roll_link').id for side in ('left','right')]
    for pose in poses:
        data.qpos[:]=pose;mujoco.mj_kinematics(native,data)
        points.append([data.xpos[i]+data.xmat[i].reshape(3,3)@np.asarray(t['target_point']) for i,t in zip(ids,tasks)])
        feet.append(data.xpos[foot_ids].copy())
    points=np.asarray(points);feet=np.asarray(feet);root=poses[:,:3]
    wanted=original29['source_task_position_w'][frames];wanted_root=original29['source_qpos29'][frames,:3]
    wanted_yaw=yaw(original29['source_qpos29'][frames,3:7]);actual_yaw=yaw(poses[:,3:7]);dy=actual_yaw-wanted_yaw
    ref_root=motion['body_pos_w'][frames,0];ref_feet=motion['body_pos_w'][frames][:,[6,12]]
    error=poses[:,7:]-motion['joint_pos'][frames]
    return dict(source_controls=len(poses),requested_source_controls=819,
        original_root_world_p95_m=float(np.percentile(np.linalg.norm(root-wanted_root,axis=1),95)),
        original_root_yaw_abs_p95_deg=float(np.percentile(np.abs(np.rad2deg(np.arctan2(np.sin(dy),np.cos(dy)))),95)),
        original_hand_head_relative_p95_m=np.percentile(np.linalg.norm((points-root[:,None])-(wanted-wanted_root[:,None]),axis=-1),95,axis=0).tolist(),
        original_hand_head_world_p95_m=np.percentile(np.linalg.norm(points-wanted,axis=-1),95,axis=0).tolist(),
        world_axis_relative_foot_p95_m=np.percentile(np.linalg.norm((feet-root[:,None])-(ref_feet-ref_root[:,None]),axis=-1),95,axis=0).tolist(),
        leg_rmse_rad=float(np.sqrt(np.mean(error[:,:12]**2))),arm_rmse_rad=float(np.sqrt(np.mean(error[:,13:]**2))),
        joint_rmse_by_hardware_joint=np.sqrt(np.mean(error**2,axis=0)).tolist())
