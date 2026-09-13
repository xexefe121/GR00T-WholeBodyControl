"""Sensor-stage, replay, heading-gauge, and stale-data checks for runtime odometry."""
from pathlib import Path
import json
import sys
import mujoco
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from gear_sonic.utils.g1_true23_bfm_imu_odometry import Native23IMUOdometry
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model,_quaternion_multiply
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL,PHYSICS


def run():
    base=Path(__file__).resolve().parent
    _,model,physics=prepare_true23_model(ROOT.parent/'GR00T-WholeBodyControl'/MODEL,ROOT/PHYSICS)
    case=base/'bfm_observable_smoke_v2/walk002_ideal'
    with np.load(case/'sensor_only_500hz.npz') as a:sensors={k:a[k].copy() for k in a.files}
    with np.load(case/'estimator_500hz.npz') as a:recorded={k:a[k].copy() for k in a.files}
    estimator=Native23IMUOdometry(model);rotated=Native23IMUOdometry(model)
    yaw=1.2;heading=np.array([np.cos(yaw/2),0.,0.,np.sin(yaw/2)])
    replay_error=heading_error=0.
    for i in range(len(sensors['joint_q'])):
        packet={k:v[i].copy() if isinstance(v[i],np.ndarray) else float(v[i]) for k,v in sensors.items()}
        result=estimator.update(**packet)
        rotated_packet={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in packet.items()}
        rotated_packet['imu_quat_wxyz']=_quaternion_multiply(heading,rotated_packet['imu_quat_wxyz'])
        other=rotated.update(**rotated_packet)
        for k,v in result.items():replay_error=max(replay_error,float(np.abs(np.asarray(v)-recorded[k][i]).max()))
        heading_error=max(heading_error,float(np.max(np.abs(result['position_start']-other['position_start']))))
    assert replay_error==0.
    assert heading_error<1e-10,heading_error
    stale_packet=packet.copy();stale_packet['timestamp_s']+=.2
    try:estimator.update(**stale_packet)
    except ValueError:stale_rejected=True
    else:stale_rejected=False
    assert stale_rejected
    # The gyro produced by mj_step must match the copied PRE-integration qvel.
    # Use a free-falling pose away from the floor with nonzero base rotation.
    probe=mujoco.MjData(model)
    probe.qpos[:3]=[0.,0.,2.];probe.qpos[3:7]=[1.,0.,0.,0.]
    probe.qpos[7:]=sensors['joint_q'][0]
    probe.qvel[3:6]=[.3,-.2,.1]
    gyro_id=model.sensor('imu-pelvis-angular-velocity').id
    accel_id=model.sensor('imu-pelvis-linear-acceleration').id
    gyro_slice=slice(model.sensor_adr[gyro_id],model.sensor_adr[gyro_id]+3)
    gyro_error=0.
    for i in range(100):
        pre=probe.qvel[3:6].copy()
        mujoco.mj_step(model,probe)
        gyro_error=max(gyro_error,float(np.abs(probe.sensordata[gyro_slice]-pre).max()))
    assert gyro_error<1e-10,gyro_error
    result=dict(sensor_only_replay_samples=len(sensors['joint_q']),sensor_only_replay_max_error=replay_error,
        initial_global_yaw_gauge_invariance_position_error_m=heading_error,stale_sample_rejected=stale_rejected,
        gyro_preintegration_timing_max_error_rad_s=gyro_error,
        imu_sensor_types=[int(model.sensor_type[gyro_id]),int(model.sensor_type[accel_id])],
        physics_hz=1/physics.timestep_s,controller_state_lag_s=physics.timestep_s,
        estimator_ground_truth_input=False,hardware_connected=False)
    (base/'bfm_runtime_odometry_checks_v1.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':run()
