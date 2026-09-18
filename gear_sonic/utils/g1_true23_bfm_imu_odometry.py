"""Unqualified native23 IMU/kinematic odometry, with no robot transport.

Inputs are timestamped joint q/dq, pelvis IMU WXYZ orientation, body gyro,
and body specific force at the known pelvis IMU site. No measured world
position, world velocity, simulator contact, force, or motion reference input.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import mujoco
import numpy as np
from gear_sonic.utils.g1_true23_step1b_mujoco import _quaternion_matrix, _quaternion_multiply


@dataclass(frozen=True)
class IMUOdometryConfig:
    enter_height_m: float = .012
    exit_height_m: float = .020
    enter_speed_m_s: float = .30
    exit_speed_m_s: float = .40
    height_sigma_m: float = .0075
    support_speed_sigma_m_s: float = .12
    velocity_observation_sigma_m_s: float = .025
    acceleration_process_sigma_m_s2: float = .20
    bias_random_walk_sigma: float = .001
    initial_velocity_sigma_m_s: float = .02
    initial_accel_bias_sigma_m_s2: float = .10
    angular_acceleration_filter_tau_s: float = .020


def finite_vector(value, width, name):
    result=np.asarray(value,dtype=np.float64)
    if result.shape!=(width,) or not np.isfinite(result).all():
        raise ValueError(f'{name} must be finite shape ({width},)')
    return result.copy()


class Native23IMUOdometry:
    """Velocity/bias Kalman filter with gated kinematic support updates.

    Initial local XY and velocity are zero. Initialization therefore assumes a
    stationary supported robot on a flat floor. Global position/yaw are not
    observable here; the initial IMU yaw defines the persistent local frame.
    """
    def __init__(self,model,config=None):
        if (model.nq,model.nv,model.nu)!=(30,29,23):
            raise ValueError('odometry requires exact native23 morphology')
        self.model=model
        self.config=config or IMUOdometryConfig()
        self.data=mujoco.MjData(model)  # private FK scratch, never physical state
        self.gravity=np.asarray(model.opt.gravity).copy()
        site=model.site('imu_in_pelvis').id
        self.imu_offset=np.asarray(model.site_pos[site]).copy()
        np.testing.assert_allclose(model.site_quat[site],[1.,0.,0.,0.],atol=1e-12)
        self.point_bodies=[];self.points=[];self.radii=[]
        for side in ('left','right'):
            body=model.body(f'{side}_ankle_roll_link').id
            ids=[i for i in range(model.ngeom) if model.geom_bodyid[i]==body
                 and model.geom_type[i]==mujoco.mjtGeom.mjGEOM_SPHERE and model.geom_contype[i]!=0]
            if len(ids)!=4:raise ValueError('four sole points required per foot')
            for i in ids:
                self.point_bodies.append(body);self.points.append(model.geom_pos[i].copy());self.radii.append(model.geom_size[i,0])
        self.radii=np.asarray(self.radii)
        self.jacp=np.zeros((3,model.nv));self.jacr=np.zeros_like(self.jacp)
        self.x=np.zeros(6)
        c=self.config
        self.covariance=np.diag([c.initial_velocity_sigma_m_s**2]*3+[c.initial_accel_bias_sigma_m_s2**2]*3)
        self.position=np.zeros(3)
        self.previous_gyro=None;self.filtered_alpha=np.zeros(3)
        self.previous_active=np.zeros(8,bool)
        self.weights=np.zeros(8);self.quaternion_start=None
        self.initial_heading=None;self.timestamp=None
        self.updates=0;self.no_contact_updates=0

    def update(self,*,timestamp_s,joint_q,joint_dq,imu_quat_wxyz,gyro_body,accel_specific_force_body):
        q=finite_vector(joint_q,23,'joint q');dq=finite_vector(joint_dq,23,'joint dq')
        quat=finite_vector(imu_quat_wxyz,4,'IMU quaternion')
        omega=finite_vector(gyro_body,3,'body gyro')
        specific_sensor=finite_vector(accel_specific_force_body,3,'specific force')
        timestamp=float(timestamp_s)
        if not np.isfinite(timestamp):raise ValueError('timestamp must be finite')
        norm=np.linalg.norm(quat)
        if norm<1e-8:raise ValueError('zero quaternion')
        quat/=norm
        first=self.timestamp is None
        dt=0. if first else timestamp-self.timestamp
        if not first and not 0.<dt<=.050:
            raise ValueError('nonmonotonic or stale odometry sample')
        if first:
            rotation=_quaternion_matrix(quat)
            yaw=np.arctan2(rotation[1,0],rotation[0,0])
            self.initial_heading=np.array([np.cos(yaw/2),0.,0.,-np.sin(yaw/2)])
        self.quaternion_start=_quaternion_multiply(self.initial_heading,quat)
        rotation=_quaternion_matrix(self.quaternion_start)
        d=self.data
        d.qpos[:3]=0.;d.qpos[3:7]=self.quaternion_start;d.qpos[7:]=q
        d.qvel[:3]=0.;d.qvel[3:6]=omega;d.qvel[6:]=dq
        mujoco.mj_kinematics(self.model,d);mujoco.mj_comPos(self.model,d)
        offsets=np.empty((8,3));relative_velocity=np.empty((8,3))
        for i,(body,point) in enumerate(zip(self.point_bodies,self.points)):
            offsets[i]=d.xpos[body]+d.xmat[body].reshape(3,3)@point
            mujoco.mj_jac(self.model,d,self.jacp,self.jacr,offsets[i],body)
            relative_velocity[i]=self.jacp@d.qvel
        if first:self.position[2]=-np.min(offsets[:,2]-self.radii)
        previous_velocity=self.x[:3].copy()
        c=self.config
        if not first:
            alpha=(omega-self.previous_gyro)/dt
            mix=dt/(c.angular_acceleration_filter_tau_s+dt)
            self.filtered_alpha=(1-mix)*self.filtered_alpha+mix*alpha
            lever=np.cross(self.filtered_alpha,self.imu_offset)+np.cross(omega,np.cross(omega,self.imu_offset))
            acceleration=rotation@(specific_sensor-lever-self.x[3:])+self.gravity
            self.x[:3]+=acceleration*dt
            transition=np.eye(6);transition[:3,3:]=-rotation*dt
            process=np.diag([c.acceleration_process_sigma_m_s2**2*dt**2]*3+[c.bias_random_walk_sigma**2*dt]*3)
            self.covariance=transition@self.covariance@transition.T+process
        bottom=offsets[:,2]-self.radii
        height=bottom-bottom.min()
        speed=np.linalg.norm(relative_velocity+self.x[:3],axis=-1)
        active=(height<=np.where(self.previous_active,c.exit_height_m,c.enter_height_m)) & (
            speed<=np.where(self.previous_active,c.exit_speed_m_s,c.enter_speed_m_s))
        self.weights[:]=0.
        if active.any():
            weights=np.exp(-(height/c.height_sigma_m)**2-(speed/c.support_speed_sigma_m_s)**2)*active
            if weights.sum()>1e-20:
                weights/=weights.sum();self.weights=weights
                measured_velocity=np.sum(-relative_velocity*weights[:,None],axis=0)
                scatter=np.sum(weights*np.sum((-relative_velocity-measured_velocity)**2,axis=-1))
                observed_cov=np.eye(3)*(c.velocity_observation_sigma_m_s**2+scatter)
                h=np.c_[np.eye(3),np.zeros((3,3))]
                gain=np.linalg.solve(h@self.covariance@h.T+observed_cov,h@self.covariance).T
                self.x+=gain@(measured_velocity-self.x[:3])
                correction=np.eye(6)-gain@h
                self.covariance=correction@self.covariance@correction.T+gain@observed_cov@gain.T
            else:active[:]=False
        if not active.any():self.no_contact_updates+=1
        if not first:self.position+=.5*(previous_velocity+self.x[:3])*dt
        self.previous_active=active;self.previous_gyro=omega;self.timestamp=timestamp;self.updates+=1
        if not np.isfinite(np.r_[self.x,self.position]).all():raise ValueError('nonfinite odometry estimate')
        return self.snapshot()

    def snapshot(self):
        return dict(position_start=self.position.copy(),velocity_start=self.x[:3].copy(),
                    accel_bias_body=self.x[3:].copy(),quaternion_start=self.quaternion_start.copy(),
                    support_weights=self.weights.copy(),timestamp_s=self.timestamp)

    def contract(self):
        return dict(kind='native23_imu_kinematic_odometry_candidate_v1',config=asdict(self.config),
                    imu_offset_body_m=self.imu_offset.tolist(),measured_root_position_input=False,
                    measured_root_velocity_input=False,actual_contact_force_input=False,
                    initial_local_xy=[0.,0.],initial_velocity_assumption=[0.,0.,0.],
                    initial_heading='pelvis IMU yaw registration once',
                    initialized_height='lower modeled sole on flat floor',
                    estimator_qualified=False,hardware_authorized=False)
