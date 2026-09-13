"""Experimental received controller with the existing native benchmark PD gains.

At each measured control boundary, convert the factory command to an equal
unclipped torque under the benchmark gains. The substep feedback law changes;
equal instantaneous torque is not a stability or full-motion guarantee.
"""
import numpy as np


class NativeBenchmarkGains:
    def __init__(self,base,contract):
        self.base=base;self.receiver=base.receiver
        self.original_kp=np.asarray(base.kp,np.float64).copy()
        self.original_kd=np.asarray(base.kd,np.float64).copy()
        self.kp=np.asarray(contract['kp'],np.float64).copy()
        self.kd=np.asarray(contract['kd'],np.float64).copy()
        self.limits=np.asarray(contract['joint_limits'],np.float64)
        if self.kp.shape!=(23,) or np.any(self.kp<=0) or np.any(self.original_kp<=0):
            raise ValueError('Both gain contracts must contain23 positive stiffnesses')
        self.previous_actual=None

    def factory_equivalent(self,target,qpos,qvel):
        q,v=np.asarray(qpos)[7:],np.asarray(qvel)[6:]
        return q+(self.kp*(np.asarray(target)-q)+(self.original_kd-self.kd)*v)/self.original_kp

    def benchmark_equivalent(self,target,qpos,qvel):
        q,v=np.asarray(qpos)[7:],np.asarray(qvel)[6:]
        return q+(self.original_kp*(np.asarray(target)-q)+(self.kd-self.original_kd)*v)/self.kp

    def receive(self,*args,**kwargs):return self.base.receive(*args,**kwargs)

    def command(self,qpos,qvel,now):
        from gear_sonic.utils.g1_true23_causal_controller import ControllerCommand
        if self.previous_actual is not None:
            equivalent=self.factory_equivalent(self.previous_actual,qpos,qvel)
            self.base.policy.previous[:]=equivalent[:12]-self.base.policy.default[:12]
        command=self.base.command(qpos,qvel,now)
        raw=self.benchmark_equivalent(command.targets,qpos,qvel)
        target=np.clip(raw,self.limits[:,0]+.06,self.limits[:,1]-.06)
        status=dict(command.status)
        status['native_benchmark_gains']=dict(enabled=True,qualified=False,
            clipped_targets=int(np.count_nonzero(raw!=target)),
            boundary_torque_change_from_clipping_max=float(np.max(np.abs(self.kp*(target-raw)))),
            received_history='actual applied native targets',
            factory_action_history='instantaneous factory-gain equivalent of actual applied target')
        return ControllerCommand(target,status)

    def commit_applied(self,qpos,qvel,target):
        self.previous_actual=np.asarray(target,np.float64).copy()
        # Received robot/command history must record what the plant received.
        # Only the separate frozen factory-policy action memory is transformed.
        self.base.commit_applied(qpos,qvel,self.previous_actual)
        equivalent=self.factory_equivalent(self.previous_actual,qpos,qvel)
        self.base.policy.previous[:]=equivalent[:12]-self.base.policy.default[:12]

    def rearm(self,now,measured_qpos):return self.base.rearm(now,measured_qpos)

    def description(self):
        return dict(kp=self.kp.tolist(),kd=self.kd.tolist(),
            original_kp=self.original_kp.tolist(),original_kd=self.original_kd.tolist(),
            gains_source='existing native23 benchmark contract, also used by the successful prepared expert',
            boundary_torque_equivalent_before_target_clipping=True,
            received_history_contains_actual_targets=True,
            model_parameters_changed=False,independent_clock_supported=False,hardware_commands=False)
