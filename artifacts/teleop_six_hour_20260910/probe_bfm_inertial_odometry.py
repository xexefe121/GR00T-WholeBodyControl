"""Offline accelerometer-aided odometry experiment; not deployment qualified.

Accelerometer EMULATION uses recorded world velocity differences, orientation,
and angular velocity. This privileged process writes declared synthetic sensor
files. The separate estimator consumes only that sensor file and fixed geometry.
It receives neither root position nor root velocity. Initial velocity is assumed
zero because these saved cases start in a stationary supported-idle pose.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
from probe_bfm_foot_odometry import (ROOT, MODEL, PHYSICS, prepare_true23_model,
    export_sensors, sensor_kinematics, _quaternion_matrix, multiply_quaternion, sha)


@dataclass(frozen=True)
class Settings:
    dt: float = .020
    enter_height: float = .012
    exit_height: float = .020
    enter_speed: float = .30
    exit_speed: float = .40
    height_sigma: float = .0075
    support_speed_sigma: float = .12
    velocity_observation_sigma: float = .025
    acceleration_process_sigma: float = .20
    bias_random_walk_sigma: float = .001
    initial_velocity_sigma: float = .02
    initial_accel_bias_sigma: float = .10
    angular_acceleration_filter: float = .5


IMU_OFFSET=np.array([.04525,0.,-.08339])
GRAVITY=np.array([0.,0.,-9.81])


def emulate_sensors(case, dt):
    """Declared ideal sensor emulation; no root position is read here."""
    sensors=export_sensors(case/'trace.npz')
    with np.load(case/'trace.npz',allow_pickle=False) as trace:
        world_velocity=trace['qvel'][:,:3].copy()
    acceleration=np.gradient(world_velocity,dt,axis=0,edge_order=2)
    angular_acceleration=np.gradient(sensors['gyro_body'],dt,axis=0,edge_order=2)
    rotations=np.asarray([_quaternion_matrix(q) for q in sensors['imu_quat_wxyz']])
    root_specific_force=np.einsum('nji,nj->ni',rotations,acceleration-GRAVITY)
    lever=(np.cross(angular_acceleration,IMU_OFFSET)+np.cross(sensors['gyro_body'],np.cross(sensors['gyro_body'],IMU_OFFSET)))
    sensors['accel_specific_force_body']=root_specific_force+lever
    return sensors


def perturb_sensors(sensors, dt, seed=260911):
    """One fixed stress case; these are declared synthetic, not measured noises."""
    rng=np.random.default_rng(seed)
    result={k:v.copy() for k,v in sensors.items()}
    n=len(result['joint_q'])
    result['joint_q']+=rng.normal(0.,.0005,(n,23))+rng.normal(0.,.003,(1,23))
    result['joint_dq']+=rng.normal(0.,.010,(n,23))
    result['gyro_body']+=np.array([.001,-.001,.002])+rng.normal(0.,.003,(n,3))
    result['accel_specific_force_body']+=np.array([.03,-.02,.02])+rng.normal(0.,.10,(n,3))
    rotation_noise=Rotation.from_rotvec(rng.normal(0.,np.deg2rad(.1),(n,3))).as_quat(scalar_first=True)
    for i in range(n):
        yaw=np.deg2rad(.05)*i*dt
        heading=np.array([np.cos(yaw/2),0.,0.,np.sin(yaw/2)])
        result['imu_quat_wxyz'][i]=multiply_quaternion(heading,multiply_quaternion(sensors['imu_quat_wxyz'][i],rotation_noise[i]))
    return result


def estimate(sensors, kin, settings):
    """Causal velocity/bias KF, gated kinematic support, position integration."""
    n=len(sensors['joint_q'])
    rotation=np.asarray([_quaternion_matrix(q) for q in sensors['imu_quat_wxyz']])
    yaw0=kin['initial_imu_yaw']
    align=np.array([[np.cos(yaw0),np.sin(yaw0),0.],[-np.sin(yaw0),np.cos(yaw0),0.],[0.,0.,1.]])
    rotation=align[None]@rotation
    offset,relvel=kin['point_offset_start'],kin['point_relative_velocity_start']
    x=np.zeros(6)  # velocity in start frame, accelerometer bias in body frame
    covariance=np.diag([settings.initial_velocity_sigma**2]*3+[settings.initial_accel_bias_sigma**2]*3)
    pos=np.zeros((n,3)); vel=np.zeros((n,3)); bias=np.zeros((n,3)); pure=np.zeros((n,3))
    inertial_velocity=np.zeros((n,3)); corrected_acc=np.zeros((n,3))
    weights_log=np.zeros((n,8)); active_log=np.zeros((n,8),bool)
    rejected=np.zeros(n,bool); innovation_log=np.zeros((n,3))
    filtered_alpha=np.zeros(3); previous_active=np.zeros(8,bool)
    h=np.c_[np.eye(3),np.zeros((3,3))]
    for t in range(n):
        r=rotation[t]
        omega=sensors['gyro_body'][t]
        if t:
            alpha=(omega-sensors['gyro_body'][t-1])/settings.dt
            filtered_alpha=(1-settings.angular_acceleration_filter)*filtered_alpha+settings.angular_acceleration_filter*alpha
        lever=np.cross(filtered_alpha,IMU_OFFSET)+np.cross(omega,np.cross(omega,IMU_OFFSET))
        specific=sensors['accel_specific_force_body'][t]-lever
        world_acc=r@(specific-x[3:])+GRAVITY
        corrected_acc[t]=world_acc
        previous_velocity=x[:3].copy()
        if t:
            x[:3]+=world_acc*settings.dt
            transition=np.eye(6);transition[:3,3:]=-r*settings.dt
            process=np.diag([settings.acceleration_process_sigma**2*settings.dt**2]*3+
                            [settings.bias_random_walk_sigma**2*settings.dt]*3)
            covariance=transition@covariance@transition.T+process
            inertial_velocity[t]=inertial_velocity[t-1]+(r@specific+GRAVITY)*settings.dt
            pure[t]=pure[t-1]+.5*(inertial_velocity[t-1]+inertial_velocity[t])*settings.dt
        bottom=offset[t,:,2]-kin['sphere_radii']
        heights=bottom-bottom.min()
        speed=np.linalg.norm(relvel[t]+x[:3],axis=-1)
        height_limit=np.where(previous_active,settings.exit_height,settings.enter_height)
        speed_limit=np.where(previous_active,settings.exit_speed,settings.enter_speed)
        active=(heights<=height_limit)&(speed<=speed_limit)
        if active.any():
            w=np.exp(-(heights/settings.height_sigma)**2-(speed/settings.support_speed_sigma)**2)*active
            if w.sum()<1e-20:
                active[:]=False
                rejected[t]=True
            else:
                w/=w.sum()
                measurement=np.sum(-relvel[t]*w[:,None],axis=0)
                scatter=np.sum(w*np.sum(((-relvel[t])-measurement)**2,axis=-1))
                observation_cov=np.eye(3)*(settings.velocity_observation_sigma**2+scatter)
                innovation=measurement-x[:3]
                s=h@covariance@h.T+observation_cov
                gain=np.linalg.solve(s,h@covariance).T
                x+=gain@innovation
                eye_minus=np.eye(6)-gain@h
                covariance=eye_minus@covariance@eye_minus.T+gain@observation_cov@gain.T
                weights_log[t]=w
                innovation_log[t]=innovation
        else:
            rejected[t]=True
        vel[t]=x[:3];bias[t]=x[3:]
        if t:
            pos[t]=pos[t-1]+.5*(previous_velocity+vel[t])*settings.dt
        active_log[t]=active
        previous_active=active
    return dict(estimated_position_start=pos,estimated_velocity_start=vel,estimated_accel_bias_body=bias,
                estimated_acceleration_start=corrected_acc,inertial_only_position_start=pure,
                inertial_only_velocity_start=inertial_velocity,inferred_contact_points=active_log,
                support_weights=weights_log,no_kinematic_update=rejected,velocity_innovation=innovation_log)


def score_only(case,kin,result,settings):
    with np.load(case/'trace.npz',allow_pickle=False) as trace:
        pos=trace['qpos'][:,:3].copy();velocity=trace['qvel'][:,:3].copy()
    yaw=kin['initial_imu_yaw']
    align=np.array([[np.cos(yaw),np.sin(yaw),0.],[-np.sin(yaw),np.cos(yaw),0.],[0.,0.,1.]])
    truth=(pos-pos[0])@align.T;true_v=velocity@align.T
    error=result['estimated_position_start']-truth
    xy=np.linalg.norm(error[:,:2],axis=-1)
    pure_error=result['inertial_only_position_start']-truth
    speed_error=np.linalg.norm((result['estimated_velocity_start']-true_v)[:,:2],axis=-1)
    weighted_stance_violation=np.sum((true_v[:,None,:]+kin['point_relative_velocity_start'])*result['support_weights'][:,:,None],axis=1)
    metrics=dict(duration_s=(len(truth)-1)*settings.dt,xy_error_p95_m=float(np.percentile(xy,95)),
                 xy_error_max_m=float(xy.max()),xy_error_final_m=float(xy[-1]),xy_error_rmse_m=float(np.sqrt(np.mean(xy**2))),
                 xy_velocity_rmse_m_s=float(np.sqrt(np.mean(speed_error**2))),
                 inertial_only_xy_final_m=float(np.linalg.norm(pure_error[-1,:2])),
                 no_kinematic_update_fraction=float(result['no_kinematic_update'].mean()),
                 accel_bias_final_m_s2=result['estimated_accel_bias_body'][-1].tolist(),
                 actual_initial_velocity_m_s_for_scoring=velocity[0].tolist(),
                 selected_stance_violation_p95_m_s=float(np.percentile(np.linalg.norm(weighted_stance_violation,axis=-1),95)))
    return metrics,dict(ground_truth_relative_start_for_score=truth,ground_truth_velocity_start_for_score=true_v,error_position_start_for_score=error)


def plot(case, output, variants, settings):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    fig.suptitle(f'{case.name}: synthetic accelerometer + joints/IMU (UNQUALIFIED)')
    for col,(label,values) in enumerate(variants.items()):
        result,scoring=values
        truth=scoring['ground_truth_relative_start_for_score'];t=np.arange(len(truth))*settings.dt
        p=result['estimated_position_start'];err=scoring['error_position_start_for_score']
        axes[0,col].plot(truth[:,0],truth[:,1],label='Hidden truth (score only)')
        axes[0,col].plot(p[:,0],p[:,1],label='Estimated from sensors')
        axes[0,col].set(xlabel='Start-frame X (m)',ylabel='Start-frame Y (m)',title=label,aspect='equal')
        axes[0,col].legend(fontsize=8,loc='best')
        axes[1,col].plot(t,np.linalg.norm(err[:,:2],axis=-1),label='XY error')
        axes[1,col].scatter(t[result['no_kinematic_update']],np.zeros(result['no_kinematic_update'].sum()),s=2,c='red',label='Inertial propagation only')
        axes[1,col].set(xlabel='Elapsed (s)',ylabel='XY error (m)')
        axes[1,col].legend(fontsize=8,loc='upper left')
    for ax in axes.ravel():ax.grid(alpha=.2)
    fig.savefig(output/'inertial_odometry_vs_hidden_truth.png',dpi=150)
    plt.close(fig)


def run(args):
    _,model,_=prepare_true23_model(ROOT.parent/'GR00T-WholeBodyControl'/MODEL,ROOT/PHYSICS)
    settings=Settings();summaries=[]
    for case in args.cases:
        case=case.resolve();output=case/'inertial_odometry_probe_v1';output.mkdir(exist_ok=True)
        if (output/'report.json').exists():raise FileExistsError(str(output/'report.json'))
        started=time.perf_counter();sensors=emulate_sensors(case,settings.dt);reports={};plots={}
        for label,variant in [('ideal',sensors),('fixed_bias_noise',perturb_sensors(sensors,settings.dt))]:
            np.savez_compressed(output/f'{label}.synthetic_sensor_input.npz',**variant)
            kin=sensor_kinematics(model,variant)
            result=estimate(variant,kin,settings)
            metrics,scoring=score_only(case,kin,result,settings)
            np.savez_compressed(output/f'{label}.estimated_trace.npz',**result)
            np.savez_compressed(output/f'{label}.privileged_score_only.npz',**scoring)
            reports[label]=metrics;plots[label]=(result,scoring)
            print(json.dumps(dict(case=case.name,variant=label,**metrics)),flush=True)
        plot(case,output,plots,settings)
        report=dict(case=case.name,settings=asdict(settings),metrics=reports,
                    synthetic_accelerometer=True,emulator_uses_recorded_ground_truth_linear_velocity=True,
                    emulator_uses_root_position=False,
                    acceleration_emulation='central finite difference of 50Hz world root velocity; body specific force plus angular lever-arm acceleration',
                    imu_offset_body_m=IMU_OFFSET.tolist(),gravity_world_m_s2=GRAVITY.tolist(),
                    estimator_inputs=list(sensors),estimator_root_position_input=False,estimator_root_velocity_input=False,
                    initial_velocity_assumption_m_s=[0.,0.,0.],initial_xy=[0.,0.],
                    timestamp_and_latency_emulation=False,physical_sensor_model_validated=False,
                    noise_scenario=dict(seed=260911,joint_q_noise_std_rad=.0005,joint_q_offset_std_rad=.003,
                        joint_dq_noise_std_rad_s=.01,gyro_noise_std_rad_s=.003,gyro_bias_rad_s=[.001,-.001,.002],
                        accel_noise_std_m_s2=.1,accel_bias_m_s2=[.03,-.02,.02],orientation_noise_std_deg=.1,yaw_drift_deg_s=.05),
                    controller_integration=False,estimator_qualified=False,
                    source_trace_sha256=sha(case/'trace.npz'),script_sha256=sha(__file__),processing_s=time.perf_counter()-started)
        (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False));summaries.append(report)
    args.summary.resolve().write_text(json.dumps(summaries,indent=2,allow_nan=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('cases',nargs='+',type=Path);parser.add_argument('--summary',required=True,type=Path)
    run(parser.parse_args())
