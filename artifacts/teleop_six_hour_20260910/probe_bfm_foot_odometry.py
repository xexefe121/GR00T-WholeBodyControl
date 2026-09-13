"""Offline, unqualified contact-sensor-free native23 foot odometry probe.

Estimator inputs: IMU quaternion/body gyro and native23 joint q/dq only.
Ground-truth translation/velocity are read exclusively by the scorer after
estimation. No source motion, contacts, forces, or robot interface is used.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model, _quaternion_matrix


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class Parameters:
    height_enter_m: float = .025
    height_exit_m: float = .040
    speed_enter_m_s: float = .50
    speed_exit_m_s: float = .80
    height_weight_m: float = .020
    speed_weight_m_s: float = .40
    anchor_blend: float = .25
    dt_s: float = .020


def export_sensors(trace_path):
    """Explicit narrow boundary: never copy free-joint position/linear velocity."""
    with np.load(trace_path, allow_pickle=False) as trace:
        sensors = dict(imu_quat_wxyz=trace['qpos'][:, 3:7].copy(),
                       joint_q=trace['qpos'][:, 7:].copy(),
                       gyro_body=trace['qvel'][:, 3:6].copy(),
                       joint_dq=trace['qvel'][:, 6:].copy())
    if set(sensors) != {'imu_quat_wxyz', 'joint_q', 'gyro_body', 'joint_dq'}:
        raise ValueError('unexpected sensor authority')
    n = len(sensors['joint_q'])
    for key, width in [('imu_quat_wxyz', 4), ('joint_q', 23), ('gyro_body', 3), ('joint_dq', 23)]:
        if sensors[key].shape != (n, width) or not np.isfinite(sensors[key]).all():
            raise ValueError(f'invalid {key}')
    return sensors


def multiply_quaternion(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return np.array([w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
                     w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v])


def sensor_kinematics(model, sensors):
    """Scratch FK at zero translation; actual simulation MjData inaccessible."""
    data = mujoco.MjData(model)
    feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'{side}_ankle_roll_link')
            for side in ('left', 'right')]
    point_bodies, points, radii = [], [], []
    for body in feet:
        geom_ids = [i for i in range(model.ngeom)
                    if model.geom_bodyid[i] == body and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE
                    and model.geom_contype[i] != 0]
        if len(geom_ids) != 4:
            raise ValueError('expected four modeled sole spheres per foot')
        for gid in geom_ids:
            point_bodies.append(body)
            points.append(model.geom_pos[gid].copy())
            radii.append(model.geom_size[gid, 0])
    first = _quaternion_matrix(sensors['imu_quat_wxyz'][0])
    yaw0 = float(np.arctan2(first[1, 0], first[0, 0]))
    alignment = np.array([np.cos(yaw0/2), 0., 0., -np.sin(yaw0/2)])
    n = len(sensors['joint_q'])
    pos, vel = np.zeros((n, 8, 3)), np.zeros((n, 8, 3))
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for t in range(n):
        quat = multiply_quaternion(alignment, sensors['imu_quat_wxyz'][t])
        data.qpos[:3] = 0.
        data.qpos[3:7] = quat / np.linalg.norm(quat)
        data.qpos[7:] = sensors['joint_q'][t]
        data.qvel[:3] = 0.
        data.qvel[3:6] = sensors['gyro_body'][t]
        data.qvel[6:] = sensors['joint_dq'][t]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        for i, (body, point) in enumerate(zip(point_bodies, points)):
            pos[t, i] = data.xpos[body] + data.xmat[body].reshape(3, 3) @ point
            mujoco.mj_jac(model, data, jacp, jacr, pos[t, i], body)
            vel[t, i] = jacp @ data.qvel
    return dict(point_offset_start=pos, point_relative_velocity_start=vel,
                sphere_radii=np.asarray(radii), initial_imu_yaw=yaw0,
                points_body=np.asarray(points), point_body_ids=np.asarray(point_bodies))


def estimate(kinematics, params):
    """Lower-sole/slow-point hypotheses, persistent anchors, weighted support.

    Uses fixed, predeclared thresholds, no ground-truth contact fitting.
    The least-height point fallback is explicit and assumes continuous support.
    Both-feet flight and broad slip remain unresolved by this sensor subset.
    """
    offsets = kinematics['point_offset_start']
    relvel = kinematics['point_relative_velocity_start']
    radii = kinematics['sphere_radii']
    n = len(offsets)
    position, velocity = np.zeros((n, 3)), np.zeros((n, 3))
    integrated = np.zeros((n, 3))
    position[0, 2] = -(offsets[0, :, 2] - radii).min()
    integrated[0] = position[0]
    anchors = position[0] + offsets[0]
    previous_active = np.zeros(8, dtype=bool)
    active_log, weights_log = np.zeros((n, 8), bool), np.zeros((n, 8))
    fallback = np.zeros(n, bool)
    anchor_disagreement = np.zeros(n)
    touchdown_count = 0
    for t in range(n):
        previous_velocity = velocity[t-1] if t else np.zeros(3)
        heights = offsets[t, :, 2] - radii
        relative_height = heights - heights.min()
        point_speed = np.linalg.norm(relvel[t] + previous_velocity, axis=-1)
        height_limit = np.where(previous_active, params.height_exit_m, params.height_enter_m)
        speed_limit = np.where(previous_active, params.speed_exit_m_s, params.speed_enter_m_s)
        active = (relative_height <= height_limit) & (point_speed <= speed_limit)
        if not active.any():
            active[np.argmin(relative_height)] = True
            fallback[t] = True
        weights = np.exp(-np.square(relative_height / params.height_weight_m)
                         -np.square(np.minimum(point_speed, 3.) / params.speed_weight_m_s)) * active
        if weights.sum() < 1e-20:
            weights = active.astype(float)
        weights /= weights.sum()
        velocity[t] = np.sum(-relvel[t] * weights[:, None], axis=0)
        predicted = position[t-1] + .5 * (previous_velocity + velocity[t]) * params.dt_s if t else position[0]
        new_contact = active & ~previous_active
        touchdown_count += int(new_contact.sum())
        anchors[new_contact] = predicted + offsets[t, new_contact]
        candidate_position = anchors - offsets[t]
        anchor_position = np.sum(candidate_position * weights[:, None], axis=0)
        position[t] = (1-params.anchor_blend) * predicted + params.anchor_blend * anchor_position
        if t:
            integrated[t] = integrated[t-1] + .5 * (previous_velocity + velocity[t]) * params.dt_s
        anchor_disagreement[t] = np.sqrt(np.sum(weights * np.sum(np.square(candidate_position - anchor_position), axis=-1)))
        active_log[t], weights_log[t] = active, weights
        previous_active = active
    return dict(estimated_position_start=position, estimated_velocity_start=velocity,
                velocity_integral_start=integrated, inferred_contact_points=active_log,
                support_weights=weights_log, fallback_assumed_support=fallback,
                anchor_disagreement_m=anchor_disagreement,
                touchdown_point_events=np.asarray(touchdown_count))


def score_only(trace_path, results, kinematics, params):
    """Privileged scorer only: align at t=0, never best-fit trajectory to truth."""
    with np.load(trace_path, allow_pickle=False) as trace:
        ground_truth = trace['qpos'][:, :3].copy()
        gt_velocity = trace['qvel'][:, :3].copy()
    yaw = kinematics['initial_imu_yaw']
    start_rotation = np.array([[np.cos(yaw), np.sin(yaw), 0.], [-np.sin(yaw), np.cos(yaw), 0.], [0., 0., 1.]])
    gt_relative = (ground_truth - ground_truth[0]) @ start_rotation.T
    gt_velocity = gt_velocity @ start_rotation.T
    estimate_relative = results['estimated_position_start'] - results['estimated_position_start'][0]
    integral_relative = results['velocity_integral_start'] - results['velocity_integral_start'][0]
    error = estimate_relative - gt_relative
    xy = np.linalg.norm(error[:, :2], axis=-1)
    elapsed = (len(xy)-1)*params.dt_s
    metrics = dict(elapsed_s=elapsed, xy_error_rmse_m=float(np.sqrt(np.mean(xy**2))),
                   xy_error_p95_m=float(np.percentile(xy, 95)), xy_error_max_m=float(xy.max()),
                   xy_error_final_m=float(xy[-1]), error_xyz_final_m=error[-1].tolist(),
                   xyz_error_p95_m=float(np.percentile(np.linalg.norm(error, axis=-1),95)),
                   xy_velocity_rmse_m_s=float(np.sqrt(np.mean(np.sum((results['estimated_velocity_start'][:, :2]-gt_velocity[:, :2])**2, axis=-1)))),
                   velocity_integral_xy_final_m=float(np.linalg.norm((integral_relative-gt_relative)[-1,:2])),
                   fallback_assumed_support_fraction=float(results['fallback_assumed_support'].mean()),
                   two_feet_inferred_support_fraction=float((results['inferred_contact_points'][:,:4].any(-1) & results['inferred_contact_points'][:,4:].any(-1)).mean()),
                   anchor_disagreement_p95_m=float(np.percentile(results['anchor_disagreement_m'],95)),
                   touchdown_point_events=int(results['touchdown_point_events']))
    return metrics, dict(ground_truth_relative_start_for_score=gt_relative,
                         ground_truth_velocity_start_for_score=gt_velocity,
                         error_position_start_for_score=error)


def plot_results(case, output, results, scoring, params):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    truth = scoring['ground_truth_relative_start_for_score']
    est = results['estimated_position_start'] - results['estimated_position_start'][0]
    integral = results['velocity_integral_start'] - results['velocity_integral_start'][0]
    t = np.arange(len(truth)) * params.dt_s
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(f'{case.name}: offline joints + IMU foot odometry (UNQUALIFIED)')
    axes[0,0].plot(truth[:,0], truth[:,1], label='Hidden truth (scoring only)', lw=2)
    axes[0,0].plot(est[:,0], est[:,1], label='Inferred support + anchors', lw=1.5)
    axes[0,0].plot(integral[:,0], integral[:,1], label='Kinematic velocity integral', alpha=.65)
    axes[0,0].scatter([0], [0], marker='o', color='black', s=20)
    axes[0,0].set(xlabel='Start-frame X (m)', ylabel='Start-frame Y (m)', aspect='equal')
    axes[0,0].legend(fontsize=8)
    axes[0,1].plot(t, np.linalg.norm(est[:,:2]-truth[:,:2],axis=-1), label='XY error')
    axes[0,1].plot(t, np.linalg.norm(integral[:,:2]-truth[:,:2],axis=-1), label='Integral XY error', alpha=.65)
    axes[0,1].set(xlabel='Elapsed (s)', ylabel='Position error (m)')
    axes[0,1].legend()
    axes[1,0].plot(t, scoring['ground_truth_velocity_start_for_score'][:,0], label='Truth vx',alpha=.6)
    axes[1,0].plot(t, results['estimated_velocity_start'][:,0], label='Estimated vx',alpha=.6)
    axes[1,0].plot(t, scoring['ground_truth_velocity_start_for_score'][:,1], label='Truth vy',alpha=.6)
    axes[1,0].plot(t, results['estimated_velocity_start'][:,1], label='Estimated vy',alpha=.6)
    axes[1,0].set(xlabel='Elapsed (s)',ylabel='Velocity (m/s)')
    axes[1,0].legend(fontsize=8)
    axes[1,1].plot(t, results['support_weights'][:,:4].sum(-1),label='Left weight')
    axes[1,1].plot(t, results['support_weights'][:,4:].sum(-1),label='Right weight',alpha=.7)
    axes[1,1].scatter(t[results['fallback_assumed_support']], np.full(results['fallback_assumed_support'].sum(),1.1),s=2,c='red',label='Forced support guess')
    axes[1,1].set(xlabel='Elapsed (s)',ylabel='Inferred support weight',ylim=(-.05,1.15))
    axes[1,1].legend(fontsize=8)
    for ax in axes.ravel():
        ax.grid(alpha=.2)
    fig.savefig(output/'odometry_vs_hidden_truth.png',dpi=150)
    plt.close(fig)


def run(args):
    _, model, _ = prepare_true23_model(ROOT.parent/'GR00T-WholeBodyControl'/MODEL,ROOT/PHYSICS)
    params = Parameters()
    summaries=[]
    for case in args.cases:
        case = case.resolve()
        output = case/'foot_odometry_probe_v1'
        output.mkdir(exist_ok=True)
        if (output/'report.json').exists():
            raise FileExistsError(str(output/'report.json'))
        started = time.perf_counter()
        sensors = export_sensors(case/'trace.npz')
        np.savez_compressed(output/'sensor_only_input.npz', **sensors)
        kinematics = sensor_kinematics(model, sensors)
        result = estimate(kinematics,params)
        metrics, privileged_scoring = score_only(case/'trace.npz', result,kinematics,params)
        np.savez_compressed(output/'estimated_trace.npz', **result, **kinematics)
        np.savez_compressed(output/'privileged_score_only.npz', **privileged_scoring)
        plot_results(case,output,result,privileged_scoring,params)
        report=dict(case=case.name, parameters=asdict(params), metrics=metrics,
                    estimator_inputs=list(sensors), actual_contact_or_force_input=False,
                    measured_root_translation_input=False, measured_root_linear_velocity_input=False,
                    desired_reference_input=False, initialized_xy=[0.,0.],
                    heading_registration='initial IMU yaw only; no trajectory alignment',
                    contact_assumption='at least one non-slipping lower sole point; fallback explicitly logged',
                    simulated_imu_noise_added=False, hardware_calibration_validated=False,
                    estimator_qualified=False, controller_integration=False,
                    source_trace_sha256=sha(case/'trace.npz'), sensor_input_sha256=sha(output/'sensor_only_input.npz'),
                    script_sha256=sha(__file__), processing_s=time.perf_counter()-started)
        (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        summaries.append(report)
        print(json.dumps(dict(case=case.name,**metrics)),flush=True)
    (args.summary.resolve()).write_text(json.dumps(summaries,indent=2,allow_nan=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('cases',type=Path,nargs='+')
    parser.add_argument('--summary',type=Path,required=True)
    run(parser.parse_args())
