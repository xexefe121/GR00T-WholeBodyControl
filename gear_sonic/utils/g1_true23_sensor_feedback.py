"""Timestamped simulated sensors -> native23 IMU/kinematic feedback.

Root position/velocity, contacts and reference motion are not accepted inputs.
Initialization requires declared stationary support; any error latches a fault.
"""
from collections import deque
from dataclasses import asdict, dataclass
import numpy as np
from scipy.spatial.transform import Rotation
from gear_sonic.utils.g1_true23_bfm_imu_odometry import Native23IMUOdometry


@dataclass(frozen=True)
class SensorEffects:
    delay_controls: int = 0
    joint_position_std_rad: float = 0.
    joint_velocity_std_rad_s: float = 0.
    orientation_std_rad: float = 0.
    gyro_std_rad_s: float = 0.
    acceleration_std_m_s2: float = 0.
    gyro_bias_rad_s: tuple = (0., 0., 0.)
    acceleration_bias_m_s2: tuple = (0., 0., 0.)
    seed: int = 0

    def __post_init__(self):
        if type(self.delay_controls) is not int or not 0 <= self.delay_controls <= 2:
            raise ValueError('sensor delay must be 0, 20 or 40 ms at 50 Hz')
        for name in ('joint_position_std_rad','joint_velocity_std_rad_s','orientation_std_rad',
                     'gyro_std_rad_s','acceleration_std_m_s2'):
            if not np.isfinite(getattr(self,name)) or getattr(self,name)<0:
                raise ValueError('sensor standard deviations must be finite and nonnegative')
        for name in ('gyro_bias_rad_s','acceleration_bias_m_s2'):
            value=np.asarray(getattr(self,name))
            if value.shape!=(3,) or not np.isfinite(value).all():raise ValueError('invalid sensor bias')


def validate_sample(sample):
    value=np.array(sample,dtype=np.float64,copy=True)
    if value.shape!=(57,) or not np.isfinite(value).all():
        raise ValueError('sensor sample requires time,q23,dq23,IMU-WXYZ4,gyro3,specific-force3')
    if abs(np.linalg.norm(value[47:51])-1)>1e-5:raise ValueError('invalid IMU quaternion norm')
    return value


class Native23SensorFeedback:
    def __init__(self, model, *, effects=None, maximum_age_s=.05):
        self.model=model
        self.effects=effects or SensorEffects()
        if not 0<maximum_age_s<=.05:raise ValueError('sensor maximum age must be at most 50 ms')
        self.maximum_age_s=maximum_age_s
        self.estimator=Native23IMUOdometry(model)
        self.rng=np.random.default_rng(self.effects.seed)
        self.samples=deque(maxlen=self.effects.delay_controls+1)
        self.last_source_time=None
        self.fault=None
        self.initialized=False

    def _measured(self,sample):
        value=validate_sample(sample);e=self.effects
        value[1:24]+=self.rng.normal(0,e.joint_position_std_rad,23)
        value[24:47]+=self.rng.normal(0,e.joint_velocity_std_rad_s,23)
        rotation=Rotation.from_quat(value[47:51][[1,2,3,0]])
        value[47:51]=(rotation*Rotation.from_rotvec(self.rng.normal(0,e.orientation_std_rad,3))).as_quat()[[3,0,1,2]]
        value[51:54]+=np.asarray(e.gyro_bias_rad_s)+self.rng.normal(0,e.gyro_std_rad_s,3)
        value[54:57]+=np.asarray(e.acceleration_bias_m_s2)+self.rng.normal(0,e.acceleration_std_m_s2,3)
        return value

    def _estimate(self,sample):
        result=self.estimator.update(timestamp_s=sample[0],joint_q=sample[1:24],joint_dq=sample[24:47],
            imu_quat_wxyz=sample[47:51],gyro_body=sample[51:54],accel_specific_force_body=sample[54:57])
        qpos=np.r_[result['position_start'],result['quaternion_start'],sample[1:24]]
        qvel=np.r_[result['velocity_start'],sample[51:54],sample[24:47]]
        if not np.isfinite(np.r_[qpos,qvel]).all():raise ValueError('nonfinite estimated feedback')
        return qpos,qvel,dict(feedback='estimated',sensor_timestamp_s=float(sample[0]),
                             estimator_updates=self.estimator.updates,ground_truth_fallback=False)

    def initialize(self,sample,*,supported_stationary):
        """One declared calibration, before movement. Delay buffer warms on real samples."""
        try:
            if self.fault is not None or self.initialized:raise ValueError('fresh estimator required for explicit reinitialization')
            if supported_stationary is not True:raise ValueError('supported stationary initialization required')
            sample=self._measured(sample)
            gravity=Rotation.from_quat(sample[47:51][[1,2,3,0]]).apply([0.,0.,1.])
            if (abs(sample[24:47]).max()>.5 or np.linalg.norm(sample[51:54])>.1
                    or gravity[2]<np.cos(.15) or abs(np.linalg.norm(sample[54:57])-9.81)>1.):
                raise ValueError('initial sensors inconsistent with quiet supported standing')
            self.last_source_time=float(sample[0]);self.samples.append(sample)
            result=self._estimate(sample);self.initialized=True
            return result
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.fault=str(exc);raise ValueError('estimator initialization fault: '+str(exc)) from exc

    def update(self,sample,*,now):
        if self.fault is not None:raise ValueError('estimator fault latched: '+self.fault)
        try:
            if not self.initialized:raise ValueError('estimator not initialized from standing')
            sample=self._measured(sample)
            if not np.isfinite(now) or not self.last_source_time<sample[0]<=now+1e-9:
                raise ValueError('nonmonotonic or future sensor sample')
            self.last_source_time=float(sample[0]);self.samples.append(sample)
            selected=self.samples[0]
            if now-selected[0]>self.maximum_age_s+1e-9:raise ValueError('stale sensor measurement')
            if selected[0]<=self.estimator.timestamp:
                raise ValueError('sensor delay buffer must warm before physics/controller release')
            result=self._estimate(selected)
            result[2]['sensor_age_s']=float(now-selected[0])
            return result
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.fault=str(exc);raise ValueError('estimator fault: '+str(exc)) from exc

    def prime_delay(self,sample):
        """Admit an actual stationary sample during setup; never duplicate/pad time."""
        if self.fault is not None:raise ValueError('estimator fault latched: '+self.fault)
        try:
            if not self.initialized:raise ValueError('estimator unavailable')
            value=self._measured(sample)
            if value[0]<=self.last_source_time:raise ValueError('delay warmup timestamp did not advance')
            if np.max(abs(value[24:47]))>.5 or np.linalg.norm(value[51:54])>.1:
                raise ValueError('delay warmup requires stationary measured samples')
            self.last_source_time=float(value[0]);self.samples.append(value)
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.fault=str(exc);raise ValueError('estimator warmup fault: '+str(exc)) from exc

    def contract(self):
        return dict(self.estimator.contract(),effects=asdict(self.effects),update_hz=50,
                    maximum_age_s=self.maximum_age_s,fault_latched=True,ground_truth_fallback=False)
