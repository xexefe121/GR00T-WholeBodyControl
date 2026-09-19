"""Experimental sampled-data damping around an existing received controller.

The actual plant keeps its native model and limits. A fixed diagonal damping
candidate comes from free-space effective inertia at the global factory pose.
Targets compensate the changed damping at each control boundary. Between
boundaries the additional damping opposes departures from sampled velocity.
This is not a stability guarantee or a qualified hardware control law.
"""
import numpy as np
import mujoco


class BoundaryMatchedDamping:
    def __init__(self,base,model,contract):
        self.base=base;self.receiver=base.receiver
        self.kp=np.asarray(base.kp,np.float64).copy()
        self.original_kd=np.asarray(base.kd,np.float64).copy()
        self.limits=np.asarray(contract['joint_limits'],np.float64)
        query=mujoco.MjData(model)
        query.qpos[:]=np.asarray(contract['initial_qpos'])
        query.qpos[7:]=base.policy.default
        query.qvel[:]=0.
        mujoco.mj_forward(model,query)
        mass=np.empty((29,29));mujoco.mj_fullM(model,mass,query.qM)
        inverse=np.linalg.solve(mass,np.eye(29))
        self.effective_inertia=1./np.diag(inverse)[6:]
        if not np.isfinite(self.effective_inertia).all() or np.any(self.effective_inertia<=0):
            raise ValueError('Invalid native effective joint inertia')
        self.kd=np.maximum(self.original_kd,2*np.sqrt(self.kp*self.effective_inertia))
        self.velocity_to_target=(self.kd-self.original_kd)/self.kp
        self.previous_actual=None

    def receive(self,*args,**kwargs):return self.base.receive(*args,**kwargs)

    def command(self,qpos,qvel,now):
        from gear_sonic.utils.g1_true23_causal_controller import ControllerCommand
        qvel=np.asarray(qvel,np.float64)
        if self.previous_actual is not None:
            # The factory network was trained with the original gains. Map the
            # last physically applied target to an equal instantaneous torque
            # under those gains, using only this measured velocity.
            equivalent=self.previous_actual-self.velocity_to_target*qvel[6:]
            self.base.policy.previous[:]=equivalent[:12]-self.base.policy.default[:12]
        command=self.base.command(qpos,qvel,now)
        requested=command.targets+self.velocity_to_target*qvel[6:]
        target=np.clip(requested,self.limits[:,0]+.06,self.limits[:,1]-.06)
        status=dict(command.status)
        status['boundary_matched_damping']=dict(
            enabled=True,qualified=False,clipped_targets=int(np.count_nonzero(target!=requested)),
            action_history='actual targets mapped to original-gain instantaneous torque at measured boundaries',
            damping_source='2*sqrt(kp/effective_inverse_mass_diagonal) at global factory pose; never below original kd',
            future_reference_frames=0)
        return ControllerCommand(target,status)

    def commit_applied(self,qpos,qvel,target):
        self.previous_actual=np.asarray(target,np.float64).copy()
        equivalent=self.previous_actual-self.velocity_to_target*np.asarray(qvel)[6:]
        self.base.commit_applied(qpos,qvel,self.previous_actual)
        self.base.policy.previous[:]=equivalent[:12]-self.base.policy.default[:12]

    def rearm(self,now,measured_qpos):return self.base.rearm(now,measured_qpos)

    def description(self):
        return dict(kp=self.kp.tolist(),original_kd=self.original_kd.tolist(),kd=self.kd.tolist(),
            effective_inertia=self.effective_inertia.tolist(),
            target_compensation='(kd-new minus kd-original)/kp times current measured joint velocity',
            original_torque_at_control_boundary_preserved_unless_target_clipped=True,
            model_parameters_changed=False,independent_clock_supported=False,hardware_commands=False)
