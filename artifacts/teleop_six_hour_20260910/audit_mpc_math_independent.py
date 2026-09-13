"""Independent scalar-cost gradient and free-flight manual-PD dynamics witnesses."""
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from gear_sonic.utils.g1_true23_mjbatch_model import position_servo_copy
from gear_sonic.utils.g1_true23_mjbatch_mpc import Native23Tracker,load_native_bundle


def main():
    base=ROOT/'artifacts/teleop_six_hour_20260910'
    native,contract,motion,_,_=load_native_bundle(base/'mjbatch_native23_inputs_v1','walk002')
    kp,kd,effort=[np.asarray(contract[key]) for key in ('kp','kd','native_effort')]
    servo=position_servo_copy(native,kp,kd,effort)
    tracker=Native23Tracker(servo,contract,motion,horizon=2,threads=1)
    tracker.window(400)
    rng=np.random.default_rng(831)
    states=tracker.states[400:403].copy()
    states[:,7:30]+=rng.normal(size=(3,23))*.02
    states[:,30:]+=rng.normal(size=(3,29))*.03
    targets=tracker.target_reference(np.arange(2))
    tracker.linearize(states,targets)
    lx,_,lu,_=tracker.expand(states,targets)
    scratch=mujoco.MjData(native)
    source_ids=np.asarray(tracker.ids)-1
    def scalar_cost(t,state,target):
        frame=400+t
        scratch.qpos[:]=state[:30]
        mujoco.mj_kinematics(native,scratch)
        value=float(np.sum((scratch.xpos[tracker.ids]-motion['body_pos_w'][frame,source_ids])**2*tracker.position_weights))
        rotation_error=np.zeros((6,3))
        for i,body in enumerate(tracker.ids):
            mujoco.mju_subQuat(rotation_error[i],scratch.xquat[body],motion['body_quat_w'][frame,source_ids[i]])
        value+=float(np.sum(rotation_error**2*tracker.rotation_weights[:,None]))
        value+=tracker.joint_weight*float(np.sum((state[7:30]-motion['joint_pos'][frame])**2))
        value+=float(np.sum((state[30:]-tracker.states[frame,30:])**2*tracker.velocity_weights))
        value+=tracker.limit_weight*float(np.sum(np.maximum(tracker.lo+tracker.limit_margin-state[7:30],0)**2+np.maximum(state[7:30]-tracker.hi+tracker.limit_margin,0)**2))
        value+=tracker.speed_weight*float(np.sum(np.maximum(np.abs(state[36:])-.8*np.asarray(contract['native_velocity']),0)**2))
        if t<2:
            goal=np.clip(motion['joint_pos'][frame+1]+kd/kp*motion['joint_vel'][frame+1],tracker.lo,tracker.hi)
            value+=tracker.control_weight*float(np.sum((target-goal)**2))
        return value
    numeric=np.empty_like(lx)
    step=1e-5
    for t in range(3):
        target=targets[t] if t<2 else np.zeros(23)
        for j in range(58):
            plus,minus=states[t].copy(),states[t].copy()
            if j<29:
                tangent=np.zeros(29);tangent[j]=step
                mujoco.mj_integratePos(native,plus[:30],tangent,1.)
                mujoco.mj_integratePos(native,minus[:30],-tangent,1.)
            else:
                plus[30+j-29]+=step;minus[30+j-29]-=step
            numeric[t,j]=(scalar_cost(t,plus,target)-scalar_cost(t,minus,target))/(2*step)
    gradient_error=float(np.max(np.abs(numeric-lx)))
    # Avoid contact active-set discontinuities in a derivative convention witness.
    flight=tracker.states[10:13].copy();flight[:,2]+=.5
    controls=np.tile(tracker.target_reference(0),(2,1))
    A,B=tracker.linearize(flight,controls)
    def manual(state,target):
        data=mujoco.MjData(native)
        data.qpos[:],data.qvel[:]=state[:30],state[30:]
        mujoco.mj_forward(native,data)
        data.qacc_warmstart[:]=0
        for _ in range(10):
            data.ctrl[:]=np.clip(kp*(target-data.qpos[7:])-kd*data.qvel[6:],-effort,effort)
            mujoco.mj_step(native,data)
        assert data.ncon==0
        return np.r_[data.qpos,data.qvel]
    errors=[]
    for _ in range(5):
        dx=rng.normal(size=58);dx/=np.linalg.norm(dx)
        du=rng.normal(size=23);du/=np.linalg.norm(du)
        plus,minus=flight[0].copy(),flight[0].copy()
        mujoco.mj_integratePos(native,plus[:30],dx[:29],step)
        mujoco.mj_integratePos(native,minus[:30],dx[:29],-step)
        plus[30:]+=step*dx[29:];minus[30:]-=step*dx[29:]
        high=manual(plus,controls[0]+step*du);low=manual(minus,controls[0]-step*du)
        observed=np.empty(58)
        mujoco.mj_differentiatePos(native,observed[:29],2*step,low[:30],high[:30])
        observed[29:]=(high[30:]-low[30:])/(2*step)
        errors.append(float(np.max(np.abs(observed-(A[0]@dx+B[0]@du)))))
    result=dict(kind='independent_mpc_math_witness',mujoco=mujoco.__version__,
                state_cost_gradient_max_abs_error=gradient_error,manual_PD_free_flight_directional_dynamics_errors=errors,
                contact_derivative_qualification=False,source_file_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                full_body_tracking_qualified=False,hardware_authorized=False)
    (base/'mpc_independent_math_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    assert gradient_error<.002
    assert max(errors)<.002


if __name__=='__main__':
    main()
