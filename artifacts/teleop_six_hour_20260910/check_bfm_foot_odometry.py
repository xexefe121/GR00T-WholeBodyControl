"""Independent FK derivative check and score-only odometry diagnostics."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import mujoco
from probe_bfm_foot_odometry import ROOT, MODEL, PHYSICS, prepare_true23_model, sensor_kinematics


def run():
    base = Path(__file__).resolve().parent
    summary = json.loads((base/'foot_odometry_probe_summary_v1.json').read_text())
    _, model, _ = prepare_true23_model(ROOT.parent/'GR00T-WholeBodyControl'/MODEL,ROOT/PHYSICS)
    # Finite differences independently check local gyro/Jacobian conventions.
    with np.load(base/'bfm_pico_arms_v3/foot_odometry_probe_v1/sensor_only_input.npz') as archive:
        ids = np.array([0, 1500, 2100, 3200, 4400, 5000])
        sensors = {k:archive[k][ids].copy() for k in archive.files}
    original = sensor_kinematics(model,sensors)
    moved = {k:v.copy() for k,v in sensors.items()}
    epsilon=1e-7
    for i in range(1,len(ids)):
        configuration=np.r_[np.zeros(3),sensors['imu_quat_wxyz'][i],sensors['joint_q'][i]]
        velocity=np.r_[np.zeros(3),sensors['gyro_body'][i],sensors['joint_dq'][i]]
        mujoco.mj_integratePos(model,configuration,velocity,epsilon)
        moved['imu_quat_wxyz'][i]=configuration[3:7]
        moved['joint_q'][i]=configuration[7:]
    perturbed=sensor_kinematics(model,moved)
    finite_difference=(perturbed['point_offset_start'][1:]-original['point_offset_start'][1:])/epsilon
    derivative_error=float(np.abs(finite_difference-original['point_relative_velocity_start'][1:]).max())
    assert derivative_error<1e-5,derivative_error
    diagnostics=[]
    for row in summary:
        case=base/row['case']
        out=case/'foot_odometry_probe_v1'
        with np.load(out/'estimated_trace.npz') as archive:
            estimate={k:archive[k].copy() for k in archive.files}
        with np.load(out/'privileged_score_only.npz') as archive:
            score={k:archive[k].copy() for k in archive.files}
        with np.load(case/'trace.npz') as trace:
            height=trace['qpos'][:,2].copy()
        weighted_gt_velocity=np.sum((score['ground_truth_velocity_start_for_score'][:,None,:]
                                     + estimate['point_relative_velocity_start'])*estimate['support_weights'][:,:,None],axis=1)
        actual_bottom_height=height[:,None]+estimate['point_offset_start'][:,:,2]-estimate['sphere_radii']
        xy_error=np.linalg.norm(score['error_position_start_for_score'][:,:2],axis=-1)
        error_change=np.diff(xy_error)
        largest=np.argsort(error_change)[-5:][::-1]+1
        diagnostic=dict(case=row['case'],
            xy_velocity_error_equals_weighted_stance_violation_max_residual=float(np.abs(
                estimate['estimated_velocity_start']-score['ground_truth_velocity_start_for_score']+weighted_gt_velocity).max()),
            selected_point_gt_speed_p95_m_s=float(np.percentile(np.linalg.norm(weighted_gt_velocity,axis=-1),95)),
            both_feet_geometrically_clear_fraction=float((actual_bottom_height.min(-1)>.020).mean()),
            support_weight_on_geometrically_clear_points_fraction=float(np.mean(np.sum(estimate['support_weights']*(actual_bottom_height>.020),axis=-1))),
            largest_xy_error_increments=[dict(t_s=float(i*.02),error_increase_m=float(error_change[i-1]),
                xy_error_m=float(xy_error[i]),actual_min_sole_clearance_m=float(actual_bottom_height[i].min()),
                weighted_gt_stance_point_speed_m_s=float(np.linalg.norm(weighted_gt_velocity[i])),
                forced_support_guess=bool(estimate['fallback_assumed_support'][i])) for i in largest])
        diagnostics.append(diagnostic)
    result=dict(fk_jacobian_forward_difference_max_error_m_s=derivative_error,
                finite_difference_perturbation_s=epsilon,
                sensors_only=['imu_quat_wxyz','joint_q','gyro_body','joint_dq'],
                diagnostics_use_ground_truth_only_after_estimation=True,diagnostics=diagnostics)
    (base/'foot_odometry_probe_checks_v1.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    run()
