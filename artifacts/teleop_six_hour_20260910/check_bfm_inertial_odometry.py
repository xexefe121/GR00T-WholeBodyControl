"""Causality and synthetic-accelerometer convention checks; no robot I/O."""
from pathlib import Path
import json
import numpy as np
from probe_bfm_inertial_odometry import (ROOT,MODEL,PHYSICS,prepare_true23_model,
    sensor_kinematics,estimate,Settings,IMU_OFFSET,GRAVITY,_quaternion_matrix)


def run():
    base=Path(__file__).resolve().parent
    case=base/'bfm_pico_yaw_only_arms_v1';output=case/'inertial_odometry_probe_v1'
    _,model,_=prepare_true23_model(ROOT.parent/'GR00T-WholeBodyControl'/MODEL,ROOT/PHYSICS)
    np.testing.assert_array_equal(model.opt.gravity,GRAVITY)
    site=model.site('imu_in_pelvis')
    np.testing.assert_allclose(model.site_pos[site.id],IMU_OFFSET,atol=1e-12,rtol=0)
    settings=Settings();checks={}
    for label in ['ideal','fixed_bias_noise']:
        with np.load(output/f'{label}.synthetic_sensor_input.npz') as archive:
            sensor={k:archive[k].copy() for k in archive.files}
        prefix={k:v[:1101].copy() for k,v in sensor.items()}
        kin=sensor_kinematics(model,prefix)
        recomputed=estimate(prefix,kin,settings)
        with np.load(output/f'{label}.estimated_trace.npz') as full:
            errors={k:float(np.abs(recomputed[k].astype(float)-full[k][:1101].astype(float)).max()) for k in recomputed}
        assert max(errors.values())==0.,errors
        checks[label]=dict(prefix_length=1101,all_estimator_arrays_prefix_difference=errors)
        if label=='ideal':
            # Emulation arithmetic check, separate from estimator causality.
            rotations=np.asarray([_quaternion_matrix(q) for q in sensor['imu_quat_wxyz']])
            gyro=sensor['gyro_body'];alpha=np.gradient(gyro,settings.dt,axis=0,edge_order=2)
            lever=np.cross(alpha,IMU_OFFSET)+np.cross(gyro,np.cross(gyro,IMU_OFFSET))
            reconstructed=np.einsum('nij,nj->ni',rotations,sensor['accel_specific_force_body']-lever)+GRAVITY
            with np.load(case/'trace.npz') as trace:
                expected=np.gradient(trace['qvel'][:,:3],settings.dt,axis=0,edge_order=2)
            maximum=float(np.abs(reconstructed-expected).max())
            assert maximum<1e-10,maximum
            checks['specific_force_emulation_reconstruction_max_error_m_s2']=maximum
    result=dict(checks=checks,causal_estimator_prefix_verified=True,
        sensor_emulation_offline_central_difference_uses_future_velocity_neighbor=True,
        future_neighbor_is_emulator_only_not_estimator_input=True,
        source_velocity_derivative_is_not_a_physical_sensor_validation=True,
        controller_integration=False)
    (base/'inertial_odometry_probe_checks_v1.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':run()
