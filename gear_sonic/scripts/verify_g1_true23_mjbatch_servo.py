"""Independent manual-PD versus affine-servo and batched-substep conformance."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from mjbatch import Batch

from gear_sonic.utils.g1_true23_mjbatch_model import position_servo_copy


def load_bundle(folder):
    model = mujoco.MjModel.from_xml_path(str(folder / "native_prepared.xml"))
    with np.load(folder / "prepared_model_arrays.npz", allow_pickle=False) as arrays:
        for name in arrays.files:
            getattr(model, name)[:] = arrays[name]
        for name in arrays.files:
            np.testing.assert_array_equal(getattr(model, name), arrays[name], name)
    mujoco.mj_setConst(model, mujoco.MjData(model))
    contract = json.loads((folder / "contract.json").read_text())
    return model, contract


def pd(target, qpos, qvel, kp, kd, effort, ranges):
    desired = np.clip(target, ranges[:, 0], ranges[:, 1])
    return np.clip(kp * (desired - qpos[7:]) - kd * qvel[6:], -effort, effort)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("choose a new servo verification output")
    native, contract = load_bundle(args.bundle)
    kp, kd, effort = [np.asarray(contract[name]) for name in ("kp", "kd", "native_effort")]
    servo = position_servo_copy(native, kp, kd, effort)
    for name in ("body_mass", "body_inertia", "body_pos", "body_quat", "jnt_range", "jnt_actfrcrange",
                 "jnt_actfrclimited", "dof_armature", "dof_damping", "dof_frictionloss", "geom_friction",
                 "geom_contype", "geom_conaffinity", "geom_pos", "geom_quat"):
        np.testing.assert_array_equal(getattr(native,name), getattr(servo,name), name)
    rng = np.random.default_rng(20260911)
    base = np.asarray(contract["initial_qpos"])
    limits = native.jnt_range[1:]
    one_step = dict(qpos=0., qvel=0., force=0.)
    saturated, clipped = 0, 0
    for _ in range(128):
        qpos = base.copy()
        qpos[7:] = np.clip(qpos[7:] + rng.uniform(-.1,.1,23), limits[:,0], limits[:,1])
        qvel = rng.uniform(-8.,8.,29)
        qvel[:6] *= .05
        raw_target = rng.uniform(limits[:,0]-.5, limits[:,1]+.5)
        target = np.clip(raw_target,limits[:,0],limits[:,1])
        torque = pd(raw_target,qpos,qvel,kp,kd,effort,limits)
        saturated += int(np.sum(np.abs(torque) >= effort-1e-12))
        clipped += int(np.sum(target != raw_target))
        a,b = mujoco.MjData(native),mujoco.MjData(servo)
        for data in (a,b):
            data.qpos[:],data.qvel[:] = qpos,qvel
        a.ctrl[:],b.ctrl[:] = torque,raw_target
        mujoco.mj_step(native,a)
        mujoco.mj_step(servo,b)
        for name,left,right in (("qpos",a.qpos,b.qpos),("qvel",a.qvel,b.qvel),("force",a.qfrc_actuator,b.qfrc_actuator)):
            one_step[name] = max(one_step[name],float(np.max(np.abs(left-right))))
    count,controls = 8,100
    batch = Batch(servo,count,num_threads=8)
    qpos,qvel,ctrl,force = [batch.bind(name) for name in ("qpos","qvel","ctrl","qfrc_actuator")]
    qpos[:] = base + np.r_[np.zeros(7),rng.uniform(-.002,.002,23)][None]
    qvel[:] = rng.uniform(-.005,.005,(count,29))
    qvel[:,:6] = 0.
    initial_q, initial_v = qpos.copy(),qvel.copy()
    targets = base[7:] + .015*np.sin(np.arange(controls)[:,None,None]*.08 + np.arange(count)[None,:,None]*.07
                                    + np.arange(23)[None,None,:]*.11)
    ctrl[:] = targets[0]
    batch.forward()
    data = [mujoco.MjData(native) for _ in range(count)]
    for i,value in enumerate(data):
        value.qpos[:],value.qvel[:] = initial_q[i],initial_v[i]
        value.ctrl[:] = pd(targets[0,i],value.qpos,value.qvel,kp,kd,effort,limits)
        mujoco.mj_forward(native,value)
    history = np.empty((count,10,batch.nstate))
    continuous = dict(qpos=0.,qvel=0.,force=0.,time=0.)
    batch_s = 0.
    for control in range(controls):
        ctrl[:] = targets[control]
        tick = time.perf_counter()
        batch.step(nstep=10,history=history)
        batch_s += time.perf_counter()-tick
        # MuJoCo integration state starts with time, qpos, qvel (this model has na=0).
        np.testing.assert_array_equal(history[:,-1,1:1+native.nq],qpos)
        np.testing.assert_array_equal(history[:,-1,1+native.nq:1+native.nq+native.nv],qvel)
        for substep in range(10):
            for i,value in enumerate(data):
                value.ctrl[:] = pd(targets[control,i],value.qpos,value.qvel,kp,kd,effort,limits)
                mujoco.mj_step(native,value)
                continuous["qpos"] = max(continuous["qpos"],float(np.max(np.abs(value.qpos-history[i,substep,1:1+native.nq]))))
                continuous["qvel"] = max(continuous["qvel"],float(np.max(np.abs(value.qvel-history[i,substep,1+native.nq:1+native.nq+native.nv]))))
                continuous["time"] = max(continuous["time"],abs(value.time-history[i,substep,0]))
        continuous["force"] = max(continuous["force"],float(np.max(np.abs(np.asarray([v.qfrc_actuator for v in data])-force))))
    tolerance = dict(qpos=1e-8,qvel=1e-6,force=1e-4,time=1e-12)
    passed = all(value <= tolerance[key] for row in (one_step,continuous) for key,value in row.items())
    report = dict(mujoco=mujoco.__version__,manual_formula="clip(kp*(clip(target,joint_bounds)-q)-kd*dq, native_effort)",
                  native23_dimensions=[native.nq,native.nv,native.nu], physical_model_fields_unchanged=True,
                  single_step_cases=128, target_clipped_joint_cases=clipped, torque_saturated_joint_cases=saturated,
                  one_step_max_abs_errors=one_step, batch_count=count, controls=controls,
                  batch_nstep=10, physics_substep_s=.002, continuous_physics_steps_per_sim=controls*10,
                  continuous_every_substep_max_abs_errors=continuous, tolerances=tolerance, numerical_parity_passed=passed,
                  batch_only_seconds=batch_s,batch_only_physics_steps_per_second=count*controls*10/batch_s,
                  dynamics_qualification_claimed=False,
                  sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
                           (args.bundle/"native_prepared.xml",args.bundle/"prepared_model_arrays.npz",args.bundle/"contract.json",Path(__file__))})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
    print(json.dumps(report,indent=2))
    if not passed:
        raise AssertionError("native affine-PD / manual-PD numerical conformance failed")


if __name__ == "__main__":
    main()
