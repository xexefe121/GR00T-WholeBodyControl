"""Private reusable native forecast, no policy/optimizer/controller connection.

Allocates MuJoCo data, state and trace workspaces once. Per-call Python result
metadata is allocated; returned array views expire on the next call. Targets
must already respect native bounds; PD effort saturation is unchanged physics.
"""
import mujoco
import numpy as np

class NativeForecast:
    def __init__(self, model, contract, maximum_controls=5):
        assert (model.nq,model.nv,model.nu,model.njnt,model.na,model.nmocap,model.nuserdata)==(30,29,23,24,0,0,0)
        assert model.opt.timestep==.002 and mujoco.__version__=='3.2.3'
        np.testing.assert_array_equal(model.actuator_trnid[:,0],np.arange(1,24))
        np.testing.assert_array_equal(model.jnt_qposadr[1:],np.arange(7,30))
        np.testing.assert_array_equal(model.jnt_dofadr[1:],np.arange(6,29))
        assert not np.any(model.actuator_dyntype) and not np.any(model.actuator_gaintype) and not np.any(model.actuator_biastype)
        np.testing.assert_array_equal(model.actuator_gainprm[:,0],np.ones(23))
        np.testing.assert_array_equal(model.actuator_gear[:,0],np.ones(23))
        np.testing.assert_array_equal(model.actuator_gear[:,1:],np.zeros((23,5)))
        self.model=model;self.spec=mujoco.mjtState.mjSTATE_INTEGRATION
        assert mujoco.mj_stateSize(model,self.spec)==291
        self.private=mujoco.MjData(model)
        self.kp,self.kd,self.effort,self.velocity=[np.asarray(contract[k],dtype=float).copy() for k in ('kp','kd','native_effort','native_velocity')]
        for a in (self.kp,self.kd,self.effort,self.velocity):assert a.shape==(23,) and np.isfinite(a).all() and np.all(a>0)
        self.limits=np.asarray(model.jnt_range[1:]).copy();self.negative_effort=-self.effort
        self.max_controls=maximum_controls;self.max_steps=maximum_controls*10
        self.before=np.empty(291);self.after=np.empty(291);self.roundtrip=np.empty(291)
        self.warning=np.empty(8,dtype=self.private.warning.number.dtype);self.lastinfo=np.empty(8,dtype=self.private.warning.lastinfo.dtype)
        self.command=np.empty(23);self.damping=np.empty(23);self.excess=np.empty(23);self.work=np.empty(23)
        self.finite_q=np.empty(30,dtype=bool);self.finite_v=np.empty(29,dtype=bool);self.finite_f=np.empty(23,dtype=bool)
        self.q=np.empty((self.max_steps+1,30));self.v=np.empty((self.max_steps+1,29));self.t=np.empty(self.max_steps+1)
        self.torque=np.empty((self.max_steps,23));self.force=np.empty((self.max_steps,23))
        self.warnings=np.empty((self.max_steps+1,8),dtype=self.warning.dtype);self.infos=np.empty_like(self.warnings)
        self.forecasts=0

    def _assess(self, step, expected_time, start_time, maximum):
        d=self.private;q,dq=d.qpos,d.qvel
        np.isfinite(q,out=self.finite_q);np.isfinite(dq,out=self.finite_v)
        finite=bool(self.finite_q.all() and self.finite_v.all())
        if finite:
            np.subtract(self.limits[:,0],q[7:],out=self.excess);np.subtract(q[7:],self.limits[:,1],out=self.work)
            np.maximum(self.excess,self.work,out=self.excess);np.maximum(self.excess,0.,out=self.excess)
            excess=float(self.excess.max())
            np.abs(dq[6:],out=self.work);np.divide(self.work,self.velocity,out=self.work);speed=float(self.work.max())
            tilt=float(np.arccos(np.clip(1-2*(q[4]**2+q[5]**2),-1,1)))
        else:
            excess=speed=tilt=float('inf');self.excess.fill(np.inf)
        np.isfinite(d.qfrc_actuator[6:],out=self.finite_f);force_finite=bool(self.finite_f.all())
        effort_ratio=0.
        if step:
            if force_finite:
                np.abs(d.qfrc_actuator[6:],out=self.work);np.divide(self.work,self.effort,out=self.work);effort_ratio=float(self.work.max())
            else:effort_ratio=float('inf')
        clock_error=abs(float(d.time)-expected_time);ideal=abs(float(d.time)-(start_time+step*.002))
        reasons=[]
        if not finite:reasons.append('nonfinite_state')
        if finite and abs(float(np.linalg.norm(q[3:7]))-1.)>1e-10:reasons.append('invalid_root_quaternion')
        if excess>1e-6:reasons.append('native_joint_bound')
        if speed>1.:reasons.append('native_joint_speed')
        if step and (not force_finite or effort_ratio>1+1e-9):reasons.append('native_actuator_effort')
        if q[2]<.25 or tilt>1.2:reasons.append('fall')
        if np.any(d.warning.number):reasons.append('engine_warning')
        if not np.isfinite(d.time) or clock_error>1e-10:reasons.append('physics_clock')
        current=dict(joint_excess_rad=excess,speed_ratio=speed,effort_ratio=effort_ratio,tilt_rad=tilt,
                     clock_error_seconds=clock_error,ideal_clock_difference_seconds=ideal)
        for key,value in current.items():maximum[key]=max(maximum[key],value)
        return None if not reasons else dict(reasons=reasons,physics_step=step,
            control_index=(step-1)//10 if step else None,substep=(step-1)%10+1 if step else 0,
            worst_joint_index=int(np.argmax(self.excess)),**current)

    def predict(self, source, target, horizon_controls):
        assert type(horizon_controls) is int and 0<horizon_controls<=self.max_controls
        target=np.asarray(target)
        assert target.shape==(23,) and target.dtype==np.float64 and np.isfinite(target).all()
        if np.any(target<self.limits[:,0]) or np.any(target>self.limits[:,1]):raise ValueError('target must already be within native bounds')
        if np.any(source.qfrc_applied) or np.any(source.xfrc_applied):raise ValueError('external forces are forbidden')
        if np.shares_memory(source.qpos,self.private.qpos) or np.shares_memory(source.qvel,self.private.qvel):raise ValueError('private data aliases caller')
        mujoco.mj_getState(self.model,source,self.before,self.spec)
        np.copyto(self.warning,source.warning.number);np.copyto(self.lastinfo,source.warning.lastinfo)
        d=self.private
        mujoco.mj_setState(self.model,d,self.before,self.spec)
        d.warning.number[:]=self.warning;d.warning.lastinfo[:]=self.lastinfo
        mujoco.mj_forward(self.model,d)
        if not np.array_equal(d.warning.number,self.warning) or not np.array_equal(d.warning.lastinfo,self.lastinfo):
            raise ValueError('private initialization produced a warning')
        mujoco.mj_setState(self.model,d,self.before,self.spec)
        d.warning.number[:]=self.warning;d.warning.lastinfo[:]=self.lastinfo
        mujoco.mj_getState(self.model,d,self.roundtrip,self.spec)
        if not np.array_equal(self.roundtrip,self.before):raise ValueError('private integration restoration mismatch')
        start=expected=float(source.time);maximum=dict(joint_excess_rad=0.,speed_ratio=0.,effort_ratio=0.,tilt_rad=0.,clock_error_seconds=0.,ideal_clock_difference_seconds=0.)
        minimum=float(d.qpos[2]);steps=0
        self.q[0]=d.qpos;self.v[0]=d.qvel;self.t[0]=d.time;self.warnings[0]=d.warning.number;self.infos[0]=d.warning.lastinfo
        try:
            first=self._assess(0,expected,start,maximum)
            if first is None:
                for step in range(1,horizon_controls*10+1):
                    np.subtract(target,d.qpos[7:],out=self.command);np.multiply(self.command,self.kp,out=self.command)
                    np.multiply(d.qvel[6:],self.kd,out=self.damping);np.subtract(self.command,self.damping,out=self.command)
                    np.maximum(self.command,self.negative_effort,out=self.command);np.minimum(self.command,self.effort,out=self.command)
                    d.ctrl[:]=self.command;mujoco.mj_step(self.model,d);expected+=.002;steps=step
                    self.q[step]=d.qpos;self.v[step]=d.qvel;self.t[step]=d.time
                    self.torque[step-1]=self.command;self.force[step-1]=d.qfrc_actuator[6:]
                    self.warnings[step]=d.warning.number;self.infos[step]=d.warning.lastinfo
                    first=self._assess(step,expected,start,maximum);minimum=min(minimum,float(d.qpos[2]))
                    if first is not None:break
        finally:
            mujoco.mj_getState(self.model,source,self.after,self.spec)
            if not np.array_equal(self.after,self.before):raise ValueError('caller integration state changed')
            if not np.array_equal(source.warning.number,self.warning) or not np.array_equal(source.warning.lastinfo,self.lastinfo):raise ValueError('caller warnings changed')
        self.forecasts+=1
        report=dict(feasible=first is None,first_failure=first,physics_steps=steps,maximum=maximum,minimum_root_height_m=minimum,
            initial_time=start,final_time=float(d.time),final_warning_counts=d.warning.number.tolist(),original_data_unchanged=True)
        trace=dict(physics_qpos=self.q[:steps+1],physics_qvel=self.v[:steps+1],physics_time=self.t[:steps+1],
            physics_torque=self.torque[:steps],physics_actuator_force=self.force[:steps],warning_counts=self.warnings[:steps+1],warning_lastinfo=self.infos[:steps+1])
        return report,trace
