"""Jacobian/dynamics audit and first-probe failure localization for WBC QP."""
import json
from pathlib import Path
import mujoco
import numpy as np
from gear_sonic.utils.g1_true23_wbc_qp import Native23WBCQP, load_bundle

root=Path("artifacts/teleop_six_hour_20260910")
model,contract,motion,original,*_=load_bundle(root/"mjbatch_native23_inputs_v1","walk002")
controller=Native23WBCQP(model,contract,motion,original)
rng=np.random.default_rng(94);checks=[]
for frame in (10,400,900):
    data=mujoco.MjData(model)
    data.qpos[:]=np.r_[motion["body_pos_w"][frame,0],motion["body_quat_w"][frame,0],motion["joint_pos"][frame]]
    data.qvel[:]=rng.normal(0,.3,29)
    mujoco.mj_forward(model,data)
    for body,local in [(controller.root_id,(0,0,0)),(controller.foot_ids[0],(.03,0,0)),(controller.hand_ids[1],(.264,.025,0))]:
        pos,jac,rot,jdotv,rdotv=controller.point(data,body,local)
        epsilon=1e-6
        sides=[]
        for sign in (-1,1):
            d=mujoco.MjData(model);d.qpos[:]=data.qpos;d.qvel[:]=data.qvel
            mujoco.mj_integratePos(model,d.qpos,data.qvel,sign*epsilon)
            mujoco.mj_forward(model,d)
            sides.append(controller.point(d,body,local))
        vel_fd=(sides[1][0]-sides[0][0])/(2*epsilon)
        jd_fd=((sides[1][1]-sides[0][1])/(2*epsilon))@data.qvel
        rd_fd=((sides[1][2]-sides[0][2])/(2*epsilon))@data.qvel
        checks.append(dict(frame=frame,body=body,jac_velocity_error=float(np.max(np.abs(vel_fd-jac@data.qvel))),
                           jdot_velocity_error=float(np.max(np.abs(jd_fd-jdotv))),
                           rotation_jdot_velocity_error=float(np.max(np.abs(rd_fd-rdotv)))))
trace=np.load(root/"wbc_qp_walk002_prefix3_v1/trace.npz")
q,v=trace["physics_qpos"][-1,7:],trace["physics_qvel"][-1,6:]
dt=.002;cfg=controller.config;w=cfg.joint_limit_barrier_frequency
upper=np.minimum.reduce([np.full(23,cfg.acceleration_max),(controller.velocity-v)/dt,
                         (controller.hi-q-dt*v)/dt**2,w*w*(controller.hi-cfg.joint_limit_margin-q)-2*w*v])
lower=np.maximum.reduce([np.full(23,-cfg.acceleration_max),(-controller.velocity-v)/dt,
                         (controller.lo-q-dt*v)/dt**2,w*w*(controller.lo+cfg.joint_limit_margin-q)-2*w*v])
joint=int(np.argmax(lower-upper))
error=trace["physics_actual_qacc"]-trace["physics_qp_qacc"]
result=dict(passed=max(max(c[k] for k in ("jac_velocity_error","jdot_velocity_error","rotation_jdot_velocity_error")) for c in checks)<1e-8,
            checks=checks,native_root_frictionloss=model.dof_frictionloss[:6].tolist(),
            failure=dict(joint=contract["joint_names"][joint],q=float(q[joint]),qvel=float(v[joint]),
                         limits=[float(controller.lo[joint]),float(controller.hi[joint])],lower_acceleration=float(lower[joint]),upper_acceleration=float(upper[joint]),
                         actual_complete_controls=len(trace["physics_torque"])//10,
                         original_v1_report_count_issue="v1 counted the next unexecuted control once;360 full controls, not361",
                         source_complete_controls=max(0,len(trace["physics_torque"])//10-350)),
            actual_minus_qp_acceleration_rms_by_dof=np.sqrt(np.mean(error**2,axis=0)).tolist(),
            actual_minus_qp_acceleration_max=float(np.max(np.abs(error))),
            model_note="v1 dynamics includes mass/bias/passive and contact forces; native free-root frictionloss0.1 is not represented in its prediction. Physical execution retains it. Contact force/compliance mismatch also remains.")
(root/"wbc_qp_numerical_audit_v2.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result))
assert result["passed"]
