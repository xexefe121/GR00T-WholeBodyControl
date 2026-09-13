"""Closed-loop native23 BFM simulation using only joint/IMU pose feedback.

No residual, hardware transport, root-force assistance, or post-init pose writes.
MuJoCo's actual pelvis-site accelerometer is sampled at 500Hz. Ground-truth
translation is retained exclusively for physical initialization and scoring.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import mujoco
import numpy as np
import torch
from gear_sonic.scripts.evaluate_g1_true23_bfmzero import ROOT,PACKAGE,load_motion,corrected_goal
from gear_sonic.utils.g1_true23_bfmzero_inference import BFMHistory,BFMZeroInference,load_contract,reference_features,state_and_terms
from gear_sonic.utils.g1_true23_bfm_imu_odometry import Native23IMUOdometry
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL,PHYSICS,task_points
from gear_sonic.utils.g1_true23_step1b_mujoco import _quaternion_matrix,_quaternion_multiply,prepare_true23_model


class SensorNoise:
    def __init__(self,enabled,seed=260911):
        self.enabled=enabled;self.rng=np.random.default_rng(seed)
        self.joint_offset=self.rng.normal(0.,.003,23)

    def apply(self,packet):
        out={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in packet.items()}
        if not self.enabled:return out
        r=self.rng
        out['joint_q']+=self.joint_offset+r.normal(0.,.0005,23)
        out['joint_dq']+=r.normal(0.,.010,23)
        out['gyro_body']+=np.array([.001,-.001,.002])+r.normal(0.,.003,3)
        out['accel_specific_force_body']+=np.array([.03,-.02,.02])+r.normal(0.,.10,3)
        rotvec=r.normal(0.,np.deg2rad(.1),3);angle=np.linalg.norm(rotvec)
        delta=np.r_[np.cos(angle/2),rotvec*np.sin(angle/2)/max(angle,1e-12)]
        yaw=np.deg2rad(.05)*packet['timestamp_s']
        heading=np.array([np.cos(yaw/2),0.,0.,np.sin(yaw/2)])
        out['imu_quat_wxyz']=_quaternion_multiply(heading,_quaternion_multiply(out['imu_quat_wxyz'],delta))
        return out


def run_case(args,clip,noise_name,policy,contract):
    output=args.output/f'{clip}_{noise_name}'
    output.mkdir(parents=True,exist_ok=False)
    _,model,physics=prepare_true23_model(ROOT.parent/'GR00T-WholeBodyControl'/MODEL,ROOT/PHYSICS)
    assert (model.nq,model.nv,model.nu)==(30,29,23)
    assert abs(physics.timestep_s-.002)<1e-12
    limits=model.jnt_range[1:].copy()
    velocity_limits=np.asarray(json.loads((ROOT/PHYSICS).read_text())['physics']['velocity_limit_hardware_radps'])
    body_ids=[model.body(n).id for n in contract['body_names']]
    gyro_id=model.sensor('imu-pelvis-angular-velocity').id
    accel_id=model.sensor('imu-pelvis-linear-acceleration').id
    gyro_slice=slice(model.sensor_adr[gyro_id],model.sensor_adr[gyro_id]+3)
    accel_slice=slice(model.sensor_adr[accel_id],model.sensor_adr[accel_id]+3)
    assert model.sensor_objid[gyro_id]==model.site('imu_in_pelvis').id
    assert model.sensor_objid[accel_id]==model.site('imu_in_pelvis').id
    motion,timeline,path=load_motion(clip)
    state,privileged=reference_features(motion,contract)
    count=len(state);available=count-11;requested=min(available,args.max_controls or available)
    data=mujoco.MjData(model)
    data.qpos[:]=np.r_[motion['body_pos_w'][10,0],motion['body_quat_w'][10,0],motion['joint_pos'][10]]
    data.qvel[:]=np.r_[motion['body_lin_vel_w'][10,0],_quaternion_matrix(data.qpos[3:7]).T@motion['body_ang_vel_w'][10,0],motion['joint_vel'][10]]
    # This test's local reference gauge must agree with the zero-XY/zero-velocity
    # estimator initialization; never inject the initial actual values into it.
    np.testing.assert_allclose(data.qpos[:2],[0.,0.],atol=1e-12)
    np.testing.assert_allclose(data.qvel[:3],[0.,0.,0.],atol=1e-12)
    mujoco.mj_forward(model,data)
    estimator=Native23IMUOdometry(model)
    noise=SensorNoise(noise_name=='fixed_bias_noise')
    packet=noise.apply(dict(timestamp_s=0.,joint_q=data.qpos[7:].copy(),joint_dq=data.qvel[6:].copy(),
        imu_quat_wxyz=data.qpos[3:7].copy(),gyro_body=data.sensordata[gyro_slice].copy(),
        accel_specific_force_body=data.sensordata[accel_slice].copy()))
    estimate=estimator.update(**packet)
    sensors={k:[v] for k,v in packet.items()}
    estimate_log={k:[v] for k,v in estimate.items()}
    heading_error0=np.arctan2(_quaternion_matrix(packet['imu_quat_wxyz'])[1,0],_quaternion_matrix(packet['imu_quat_wxyz'])[0,0])
    # Ground truth is rotated into the estimator's initial IMU frame for scoring.
    score_rotation=np.array([[np.cos(heading_error0),np.sin(heading_error0),0.],[-np.sin(heading_error0),np.cos(heading_error0),0.],[0.,0.,1.]])
    initial_gt=data.qpos[:3].copy();initial_est=estimate['position_start'].copy()
    sensor_errors=[np.zeros(3)]
    history=BFMHistory();action=np.zeros(23,np.float32)
    names=('qpos','qvel','action','target','state','history','joint_error','root_error','landmark_error',
           'relative_landmark_error','range_excess','velocity_ratio','effort_ratio','inference_ms',
           'estimated_root','feedback_qpos','estimator_age_ms')
    traces={k:[] for k in names};traces['qpos']=[data.qpos.copy()];traces['qvel']=[data.qvel.copy()]
    failure=None;started=time.perf_counter()
    for control in range(requested):
        frame=control+11
        sensed,terms=state_and_terms(packet['joint_q'],packet['joint_dq'],packet['imu_quat_wxyz'],packet['gyro_body'],action,contract['default_q'])
        hist=history.before_update(terms)
        feedback_qpos=np.r_[estimate['position_start'],estimate['quaternion_start'],packet['joint_q']]
        age_ms=(data.time-estimate['timestamp_s'])*1000
        tick=time.perf_counter()
        goal=corrected_goal(policy,state,privileged,motion,frame,feedback_qpos,8,args.position_gain,args.yaw_gain)
        raw=policy.actor(torch.from_numpy(sensed[None]),torch.from_numpy(action[None]),torch.from_numpy(hist[None]),goal)[0].numpy()
        action=raw*5.
        target=contract['default_q']+action*.25*contract['training_effort']/contract['kp']
        target[13:]=motion['joint_pos'][frame,13:]+contract['kd'][13:]/contract['kp'][13:]*motion['joint_vel'][frame,13:]
        clipped=np.clip(target,limits[:,0],limits[:,1])
        action=((target-contract['default_q'])*contract['kp']/(.25*contract['training_effort'])).astype(np.float32)
        inference_ms=(time.perf_counter()-tick)*1000
        peak_effort=peak_range=peak_velocity=0.
        for _ in range(physics.decimation):
            torque=contract['kp']*(clipped-data.qpos[7:])-contract['kd']*data.qvel[6:]
            data.ctrl[:]=np.clip(torque,-physics.effort,physics.effort)
            # mj_step computes accelerometer/gyro for this pre-integration state.
            # Pair the resulting sensor values with copies at the same timestamp.
            pre=dict(timestamp_s=float(data.time),joint_q=data.qpos[7:].copy(),joint_dq=data.qvel[6:].copy(),imu_quat_wxyz=data.qpos[3:7].copy())
            gt_for_scoring=data.qpos[:3].copy()
            mujoco.mj_step(model,data)
            if pre['timestamp_s']>estimator.timestamp+1e-10:
                pre['gyro_body']=data.sensordata[gyro_slice].copy()
                pre['accel_specific_force_body']=data.sensordata[accel_slice].copy()
                packet=noise.apply(pre);estimate=estimator.update(**packet)
                for k,v in packet.items():sensors[k].append(v)
                for k,v in estimate.items():estimate_log[k].append(v)
                sensor_errors.append((estimate['position_start']-initial_est)-score_rotation@(gt_for_scoring-initial_gt))
            peak_effort=max(peak_effort,float(np.max(np.abs(data.qfrc_actuator[6:])/physics.effort)))
            peak_range=max(peak_range,float(np.max(np.maximum(limits[:,0]-data.qpos[7:],data.qpos[7:]-limits[:,1]))))
            peak_velocity=max(peak_velocity,float(np.max(np.abs(data.qvel[6:])/velocity_limits)))
        mujoco.mj_kinematics(model,data)
        actual=task_points(data.xpos[body_ids],data.xquat[body_ids]);desired=task_points(motion['body_pos_w'][frame],motion['body_quat_w'][frame])
        root_error=data.qpos[:3]-motion['body_pos_w'][frame,0]
        values=dict(qpos=data.qpos.copy(),qvel=data.qvel.copy(),action=action.copy(),target=clipped,
            state=sensed,history=hist,joint_error=data.qpos[7:]-motion['joint_pos'][frame],root_error=root_error,
            landmark_error=np.linalg.norm(actual-desired,axis=-1),relative_landmark_error=np.linalg.norm(actual-desired-root_error,axis=-1),
            range_excess=peak_range,velocity_ratio=peak_velocity,effort_ratio=peak_effort,inference_ms=inference_ms,
            estimated_root=estimate['position_start'].copy(),feedback_qpos=feedback_qpos,estimator_age_ms=age_ms)
        for k,v in values.items():traces[k].append(v)
        tilt=float(np.arccos(np.clip(_quaternion_matrix(data.qpos[3:7])[2,2],-1,1)))
        if not np.isfinite(data.qpos).all() or data.qpos[2]<.25 or tilt>1.2:
            failure=dict(kind='fall',control=control,height=float(data.qpos[2]),tilt=tilt);break
        if peak_range>.01 or peak_velocity>1.:
            failure=dict(kind='physical_limit',control=control,range_excess=peak_range,velocity_ratio=peak_velocity);break
        if (control+1)%500==0:
            print(json.dumps(dict(clip=clip,noise=noise_name,controls=control+1,root_error=float(np.linalg.norm(root_error)),odom_xy_error=float(np.linalg.norm(sensor_errors[-1][:2])))),flush=True)
    arrays={k:np.asarray(v) for k,v in traces.items()}
    np.savez_compressed(output/'trace.npz',**arrays)
    np.savez_compressed(output/'sensor_only_500hz.npz',**{k:np.asarray(v) for k,v in sensors.items()})
    np.savez_compressed(output/'estimator_500hz.npz',**{k:np.asarray(v) for k,v in estimate_log.items()})
    np.savez_compressed(output/'privileged_odometry_score_only.npz',error_position_start=np.asarray(sensor_errors))
    completed=len(arrays['action']);phase=next(p for p in timeline['phases'] if p['name']=='source_motion')
    start,stop=phase['control_start'],min(completed,phase['control_stop']);metrics=None
    if stop>start:
        error=arrays['joint_error'][start:stop]
        metrics=dict(source_controls=stop-start,source_requested=phase['requested_controls'],
            leg_rmse=float(np.sqrt(np.mean(error[:,:12]**2))),arm_rmse=float(np.sqrt(np.mean(error[:,13:]**2))),
            root_p95=float(np.percentile(np.linalg.norm(arrays['root_error'][start:stop],axis=-1),95)),
            landmark_p95=np.percentile(arrays['landmark_error'][start:stop],95,axis=0).tolist(),
            relative_landmark_p95=np.percentile(arrays['relative_landmark_error'][start:stop],95,axis=0).tolist())
    odomxy=np.linalg.norm(np.asarray(sensor_errors)[:,:2],axis=-1)
    result=dict(clip=clip,sensor_noise=noise_name,completed=completed,requested=requested,available=available,
        failure=failure,full_lifecycle_completed=completed==available and failure is None,source_metrics=metrics,
        odometry_metrics=dict(xy_error_p95_m=float(np.percentile(odomxy,95)),xy_error_max_m=float(odomxy.max()),
            xy_error_final_m=float(odomxy[-1]),no_kinematic_update_fraction=estimator.no_contact_updates/estimator.updates,
            sensor_samples=estimator.updates,age_ms_p95=float(np.percentile(arrays['estimator_age_ms'],95))),
        range_excess_max=float(arrays['range_excess'].max()),effort_ratio_max=float(arrays['effort_ratio'].max()),
        velocity_ratio_max=float(arrays['velocity_ratio'].max()),inference_ms_p95=float(np.percentile(arrays['inference_ms'],95)),
        elapsed_seconds=time.perf_counter()-started,goal_horizon=8,goal_buffer_ms=140,reference_path=str(path),
        position_gain=args.position_gain,yaw_gain=args.yaw_gain,arm_reference=True,residual_checkpoint=None,
        ground_truth_pose_feedback=False,root_xy_feedback_source='causal IMU+joint kinematic estimator',
        sensor_contract='native MuJoCo pelvis accelerometer and gyro at500Hz, quaternion/joint encoders paired pre-integration',
        finite_difference_ground_truth_acceleration=False,estimator_contract=estimator.contract(),
        perturbation_contract=None if noise_name=='ideal' else dict(seed=260911,joint_q_offset_std_rad=.003,joint_q_noise_std_rad=.0005,
            joint_dq_noise_std_rad_s=.01,gyro_bias_rad_s=[.001,-.001,.002],gyro_noise_std_rad_s=.003,
            accel_bias_m_s2=[.03,-.02,.02],accel_noise_std_m_s2=.1,orientation_noise_std_deg=.1,yaw_drift_deg_s=.05),
        noise_scope='actor and estimator sensor packets; joint servo simulation retains its ideal internal feedback',
        model_path=str(ROOT.parent/'GR00T-WholeBodyControl'/MODEL),hardware_authorized=False,deployment_ready=False)
    (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(clip=clip,noise=noise_name,completed=completed,failure=failure,source_metrics=metrics,odometry=result['odometry_metrics'])),flush=True)
    return result


def main(args):
    torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),ROOT/'gear_sonic/utils/g1_true23_bfm_imu_odometry.py',ROOT/'gear_sonic/scripts/evaluate_g1_true23_bfmzero.py',ROOT/'gear_sonic/utils/g1_true23_bfmzero_inference.py']
    (args.output/'provenance.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
    for p in files:(args.output/p.name).write_bytes(p.read_bytes())
    contract=load_contract(PACKAGE/'bfmzero_inspect_v1/config.yaml')
    policy=BFMZeroInference(PACKAGE/'bfmzero_inference_v1/inference.safetensors')
    results=[]
    for clip in args.clips:
        for noise_name in args.noise:
            results.append(run_case(args,clip,noise_name,policy,contract))
            (args.output/'summary.json').write_text(json.dumps(results,indent=2,allow_nan=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--clips',nargs='+',choices=['walk002','walk003','walk008','pico'],default=['walk002','walk003','walk008','pico'])
    parser.add_argument('--noise',nargs='+',choices=['ideal','fixed_bias_noise'],default=['ideal','fixed_bias_noise'])
    parser.add_argument('--max-controls',type=int);parser.add_argument('--position-gain',type=float,default=1.);parser.add_argument('--yaw-gain',type=float,default=2.)
    main(parser.parse_args())
