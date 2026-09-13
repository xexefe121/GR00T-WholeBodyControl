"""Explicit original29-to-native23 torso/feet/hand intent retargeting.

Reference kinematics only. The floating pelvis replaces missing waist tilt;
both six-axis legs solve against unchanged world foot poses. No simulation
state is set by this code and no dynamic feasibility is claimed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from gear_sonic.scripts.evaluate_g1_true23_bfmzero import ROOT, DATA, MODEL, PHYSICS, load_motion
from gear_sonic.utils.g1_true23_hand_frame_tasks import neutral_wrist_hand_tasks


def kinematics(model, data):
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)


def skew(vector):
    x,y,z=vector
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


class Native23LegIK:
    def __init__(self, model, *, rotation_weight=.3, posture_weight=.0001, max_nfev=40):
        if (model.nq,model.nv,model.nu)!=(30,29,23):
            raise ValueError("requires native23 model")
        self.model=model
        self.data=mujoco.MjData(model)
        self.rotation_weight=rotation_weight
        self.posture_weight=posture_weight
        self.max_nfev=max_nfev
        self.bodies=[model.body(f"{side}_ankle_roll_link").id for side in ("left","right")]
        self.last_solution_info=None

    def residual_jacobian(self, q, side, base_pose, target_position, target_rotation, posture):
        indices=np.arange(side*6,side*6+6)
        data,model=self.data,self.model
        data.qpos[:]=base_pose
        data.qpos[7+indices]=q
        kinematics(model,data)
        body=self.bodies[side]
        rotation=data.xmat[body].reshape(3,3).copy()
        jacp,jacr=np.zeros((3,model.nv)),np.zeros((3,model.nv))
        mujoco.mj_jacBody(model,data,jacp,jacr,body)
        rotation_jac=np.stack([skew(jacr[:,6+j])@rotation for j in indices],axis=-1).reshape(9,6)
        residual=np.r_[data.xpos[body]-target_position,
                       self.rotation_weight*(rotation-target_rotation).ravel(),
                       self.posture_weight*(q-posture)]
        jacobian=np.vstack((jacp[:,6+indices],self.rotation_weight*rotation_jac,
                            self.posture_weight*np.eye(6)))
        return residual,jacobian

    def solve(self, qpos, target_positions, target_rotations, posture_qpos=None):
        result=np.asarray(qpos,dtype=np.float64).copy()
        posture=result if posture_qpos is None else np.asarray(posture_qpos)
        errors=[];angles=[];counts=[]
        for side in range(2):
            indices=np.arange(side*6,side*6+6)
            bounds=self.model.jnt_range[1+indices].copy()
            args=(side,result,target_positions[side],target_rotations[side],posture[7+indices])
            initial=np.clip(result[7+indices],bounds[:,0]+1e-9,bounds[:,1]-1e-9)
            solution=least_squares(lambda q:self.residual_jacobian(q,*args)[0],initial,
                                   jac=lambda q:self.residual_jacobian(q,*args)[1],
                                   bounds=(bounds[:,0],bounds[:,1]),max_nfev=self.max_nfev,
                                   ftol=1e-9,xtol=1e-9,gtol=1e-9)
            result[7+indices]=solution.x
            self.data.qpos[:]=result
            kinematics(self.model,self.data)
            body=self.bodies[side]
            errors.append(float(np.linalg.norm(self.data.xpos[body]-target_positions[side])))
            angles.append(float(Rotation.from_matrix(self.data.xmat[body].reshape(3,3)@target_rotations[side].T).magnitude()))
            counts.append(solution.nfev)
        return result,np.array(errors),np.array(angles),np.array(counts)

    def solve_with_height(self, qpos, target_positions, target_rotations, posture_qpos=None,
                          *, previous_qpos=None, max_step_rad=None):
        """Fit both feet with <=6cm height relief and optional causal joint-step bounds.

        The previous pose warms the coupled solve and defines a joint-specific
        step box intersected with physical ranges. Infeasible foot goals remain
        visible as residuals; source time and goals are never altered.
        """
        result=np.asarray(qpos,dtype=np.float64).copy()
        posture=result if posture_qpos is None else np.asarray(posture_qpos)
        if result.shape!=(30,) or not np.isfinite(result).all():
            raise ValueError("qpos must contain 30 finite native23 coordinates")
        if posture.shape!=(30,) or not np.isfinite(posture).all():
            raise ValueError("posture_qpos must contain 30 finite coordinates")
        if (previous_qpos is None)!=(max_step_rad is None):
            raise ValueError("previous_qpos and max_step_rad must be supplied together")
        previous=None
        step=None
        if previous_qpos is not None:
            previous=np.asarray(previous_qpos,dtype=np.float64)
            if previous.shape!=(30,) or not np.isfinite(previous).all():
                raise ValueError("previous_qpos must contain 30 finite coordinates")
            step=np.asarray(max_step_rad,dtype=np.float64)
            if step.ndim==0:
                step=np.full(12,float(step))
            if step.shape!=(12,) or not np.isfinite(step).all() or np.any(step<=0):
                raise ValueError("max_step_rad must be a positive finite scalar or 12-vector")
        desired_height=float(result[2])
        bounds=self.model.jnt_range[1:13].copy()
        def evaluate(x):
            pose=result.copy();pose[7:19]=x[:12];pose[2]=x[12]
            residuals=[];jacobians=[]
            for side in range(2):
                indices=np.arange(side*6,side*6+6)
                residual,jacobian=self.residual_jacobian(x[indices],side,pose,target_positions[side],
                                                       target_rotations[side],posture[7+indices])
                jac=np.zeros((len(residual),13))
                jac[:,indices]=jacobian
                jac[2,12]=1.
                residuals.append(residual);jacobians.append(jac)
            residuals.append(np.array([.3*(x[12]-desired_height)]))
            height_jac=np.zeros((1,13));height_jac[0,12]=.3
            jacobians.append(height_jac)
            return np.concatenate(residuals),np.vstack(jacobians)
        lower=np.r_[bounds[:,0],desired_height-.06]
        upper=np.r_[bounds[:,1],desired_height+.06]
        initial=np.r_[result[7:19],desired_height]
        if previous is not None:
            lower[:12]=np.maximum(lower[:12],previous[7:19]-step)
            upper[:12]=np.minimum(upper[:12],previous[7:19]+step)
            initial=np.r_[previous[7:19],previous[2]]
        if np.any(lower>=upper):
            raise ValueError("previous leg step box has no interior inside physical bounds")
        initial=np.clip(initial,np.nextafter(lower,upper),np.nextafter(upper,lower))
        solution=least_squares(lambda x:evaluate(x)[0],initial,jac=lambda x:evaluate(x)[1],
                               bounds=(lower,upper),max_nfev=self.max_nfev,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        self.last_solution_info=dict(success=bool(solution.success),status=int(solution.status),
                                     message=str(solution.message),cost=float(solution.cost),
                                     optimality=float(solution.optimality),nfev=int(solution.nfev))
        result[7:19]=solution.x[:12];result[2]=solution.x[12]
        self.data.qpos[:]=result
        kinematics(self.model,self.data)
        errors=[np.linalg.norm(self.data.xpos[body]-target_positions[i]) for i,body in enumerate(self.bodies)]
        angles=[Rotation.from_matrix(self.data.xmat[body].reshape(3,3)@target_rotations[i].T).magnitude()
                for i,body in enumerate(self.bodies)]
        return result,np.array(errors),np.array(angles),np.full(2,solution.nfev)


def torso_matched_base(native, data, seed, target_position, target_rotation):
    data.qpos[:]=seed
    data.qpos[:7]=[0,0,0,1,0,0,0]
    mujoco.mj_kinematics(native,data)
    torso=native.body("torso_link").id
    local_position=data.xpos[torso].copy()
    local_rotation=data.xmat[torso].reshape(3,3).copy()
    base_rotation=target_rotation@local_rotation.T
    result=seed.copy()
    result[:3]=target_position-base_rotation@local_position
    result[3:7]=Rotation.from_matrix(base_rotation).as_quat()[[3,0,1,2]]
    return result


def run(clip, output, max_frames=None):
    from gear_sonic.utils.g1_true23_intent_arm_ik import Native23ArmIK
    import gear_sonic.utils.g1_true23_intent_arm_ik as arm_module
    import gear_sonic.utils.g1_true23_hand_frame_tasks as hand_geometry_module
    if max_frames is not None and (not isinstance(max_frames,int) or max_frames<2):
        raise ValueError("max_frames must be an integer of at least two")
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    native_path=ROOT.parent/"GR00T-WholeBodyControl"/MODEL
    source_path=ROOT.parent/"GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_29dof.xml"
    native=mujoco.MjModel.from_xml_path(str(native_path))
    source=mujoco.MjModel.from_xml_path(str(source_path))
    nd,sd=mujoco.MjData(native),mujoco.MjData(source)
    legs,arms=Native23LegIK(native,posture_weight=.002),Native23ArmIK(native,posture_weight=.02)
    tasks,task_convention=neutral_wrist_hand_tasks(source,native)
    tasks=[next(t for t in tasks if t.name==name) for name in ("left_hand","right_hand","head_proxy")]
    original_path=DATA/("pico_freedancing_v1/optical_reference_v2/original29.npz" if clip=="pico" else
                        f"{clip}/original_source_bundle_v1/original_reference.npz")
    motion,timeline,native_reference_path=load_motion(clip)
    if not np.array_equal(motion["fps"],[50.]):
        raise ValueError("BFM intent export requires the unchanged 50 Hz source timeline")
    dt=.02
    physics_path=ROOT/PHYSICS
    velocity_limits=np.asarray(json.loads(physics_path.read_text())["physics"]["velocity_limit_hardware_radps"])
    leg_step=.8*velocity_limits[:12]*dt
    with np.load(original_path,allow_pickle=False) as z:
        original={key:z[key].copy() for key in z.files}
    count=len(motion["joint_pos"])
    if len(original["source_qpos29"])!=count:
        raise ValueError("original/native timelines differ")
    if max_frames:
        count=min(count,max_frames)
    source_provenance={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                       (original_path,native_reference_path,native_path,source_path,physics_path,
                        Path(__file__),Path(arm_module.__file__),Path(hand_geometry_module.__file__))}
    (output/"retarget_runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    (output/"arm_ik_snapshot.py").write_bytes(Path(arm_module.__file__).read_bytes())
    (output/"hand_geometry_snapshot.py").write_bytes(Path(hand_geometry_module.__file__).read_bytes())
    (output/"physics_contract_snapshot.json").write_bytes(physics_path.read_bytes())
    records={key:[] for key in ("qpos","body_pos_w","body_quat_w","foot_position_error",
                                "foot_orientation_error","leg_nfev","arm_nfev","original_task_error",
                                "leg_solver_success","leg_solver_status","leg_solver_cost",
                                "leg_solver_optimality","arm_solver_success","arm_orientation_error")}
    failures=[]
    torso=source.body("torso_link").id
    source_feet=[source.body(f"{side}_ankle_roll_link").id for side in ("left","right")]
    started=time.perf_counter()
    previous_qpos=None
    for frame in range(count):
        sd.qpos[:]=original["source_qpos29"][frame]
        mujoco.mj_kinematics(source,sd)
        seed=np.r_[motion["body_pos_w"][frame,0],motion["body_quat_w"][frame,0],motion["joint_pos"][frame]]
        qpos=torso_matched_base(native,nd,seed,sd.xpos[torso],sd.xmat[torso].reshape(3,3))
        foot_positions=sd.xpos[source_feet].copy()
        foot_rotations=sd.xmat[source_feet].reshape(2,3,3).copy()
        qpos,foot_error,foot_angle,leg_nfev=legs.solve_with_height(
            qpos,foot_positions,foot_rotations,seed if previous_qpos is None else previous_qpos,
            previous_qpos=previous_qpos,max_step_rad=None if previous_qpos is None else leg_step)
        if previous_qpos is not None:
            qpos[20:]=previous_qpos[20:]
        arm_result=arms.solve(qpos,original["source_task_position_w"][frame,:2],
                              target_quaternions_w=original["source_task_quaternion_wxyz"][frame,:2],
                              posture_qpos=seed if previous_qpos is None else previous_qpos,
                              previous_qpos=previous_qpos,max_step_rad=None if previous_qpos is None else .15)
        qpos=arm_result.qpos
        previous_qpos=qpos.copy()
        nd.qpos[:]=qpos
        mujoco.mj_kinematics(native,nd)
        actual_tasks=np.array([nd.xpos[native.body(t.target_body).id]+
                               nd.xmat[native.body(t.target_body).id].reshape(3,3)@t.target_point for t in tasks])
        if not legs.last_solution_info["success"] or not arm_result.success.all():
            failures.append(dict(frame=frame,leg=legs.last_solution_info.copy(),
                                 arm_success=arm_result.success.tolist(),arm_nfev=arm_result.nfev.tolist(),
                                 foot_position_error_m=foot_error.tolist(),
                                 hand_position_error_m=arm_result.position_errors_m.tolist()))
        values=dict(qpos=qpos,body_pos_w=nd.xpos[1:].copy(),body_quat_w=nd.xquat[1:].copy(),
                    foot_position_error=foot_error,foot_orientation_error=foot_angle,leg_nfev=leg_nfev,
                    arm_nfev=arm_result.nfev,original_task_error=np.linalg.norm(actual_tasks-original["source_task_position_w"][frame],axis=-1),
                    leg_solver_success=legs.last_solution_info["success"],leg_solver_status=legs.last_solution_info["status"],
                    leg_solver_cost=legs.last_solution_info["cost"],leg_solver_optimality=legs.last_solution_info["optimality"],
                    arm_solver_success=arm_result.success,arm_orientation_error=arm_result.orientation_errors_rad)
        for key,value in values.items():
            records[key].append(value)
        if (frame+1)%500==0:
            print(json.dumps(dict(clip=clip,frames=frame+1,elapsed_s=time.perf_counter()-started)),flush=True)
    arrays={key:np.asarray(value) for key,value in records.items()}
    qpos=arrays["qpos"]
    quaternions=arrays["body_quat_w"]
    world_angular=np.empty((count,24,3))
    for body in range(24):
        rotation=Rotation.from_quat(quaternions[:,body,[1,2,3,0]])
        earlier=np.maximum(np.arange(count)-1,0)
        later=np.minimum(np.arange(count)+1,count-1)
        world_angular[:,body]=(rotation[later]*rotation[earlier].inv()).as_rotvec()/((later-earlier)*dt)[:,None]
    exported=dict(fps=np.array([50.]),joint_pos=qpos[:,7:],joint_vel=np.gradient(qpos[:,7:],dt,axis=0),
                  body_pos_w=arrays["body_pos_w"],body_quat_w=quaternions,
                  body_lin_vel_w=np.gradient(arrays["body_pos_w"],dt,axis=0),body_ang_vel_w=world_angular)
    np.savez_compressed(output/"reference.npz",**exported)
    np.savez_compressed(output/"kinematic_diagnostics.npz",**arrays)
    (output/"solver_failures.json").write_text(json.dumps(failures,indent=2,allow_nan=False))
    adjacent_velocity=np.diff(qpos[:,7:],axis=0)/dt
    adjacent_ratio=np.abs(adjacent_velocity)/velocity_limits
    source_phase=next(p for p in timeline["phases"] if p["name"]=="source_motion")
    phase=slice(source_phase["frame_start"],min(count,source_phase["frame_stop"]))
    subset=arrays["original_task_error"][phase]
    report=dict(clip=clip,frames=count,full_original_timeline=count==len(motion["joint_pos"]),
                original_task_p95_m=None if len(subset)==0 else np.percentile(subset,95,axis=0).tolist(),
                foot_position_error_p95_m=np.percentile(arrays["foot_position_error"],95,axis=0).tolist(),
                foot_position_error_max_m=np.max(arrays["foot_position_error"],axis=0).tolist(),
                foot_orientation_error_p95_rad=np.percentile(arrays["foot_orientation_error"],95,axis=0).tolist(),
                joint_velocity_max_rad_s=np.max(np.abs(exported["joint_vel"]),axis=0).tolist(),
                adjacent_joint_velocity_max_rad_s=np.max(np.abs(adjacent_velocity),axis=0).tolist(),
                adjacent_velocity_ratio_max=float(adjacent_ratio.max()),
                adjacent_leg_velocity_ratio_max=float(adjacent_ratio[:,:12].max()),
                adjacent_velocity_limit_violation_count=int(np.sum(adjacent_ratio>1.+1e-9)),
                adjacent_leg_bound_violation_count=int(np.sum(adjacent_ratio[:,:12]>.8+1e-9)),
                leg_max_step_rad_per_control=leg_step.tolist(),leg_velocity_limit_fraction=.8,
                leg_temporal_posture_weight=legs.posture_weight,leg_warm_start="previous_exported_pose",
                leg_solver_failure_count=int(np.sum(~arrays["leg_solver_success"])),
                arm_solver_failure_count=int(np.sum(~arrays["arm_solver_success"])),
                solver_failure_frame_count=len(failures),
                original_task_max_m=np.max(arrays["original_task_error"],axis=0).tolist(),
                foot_orientation_error_max_rad=np.max(arrays["foot_orientation_error"],axis=0).tolist(),
                hand_orientation_error_p95_rad=np.percentile(arrays["arm_orientation_error"],95,axis=0).tolist(),
                root_shift_p95_m=float(np.percentile(np.linalg.norm(qpos[:,:3]-motion["body_pos_w"][:count,0],axis=-1),95)),
                elapsed_s=time.perf_counter()-started,kind="torso_orientation_feet_se3_bounded_height_causal_leg_arm_ik_reference_v3",
                arm_max_step_rad_per_control=.15,arm_temporal_posture_weight=.02,
                pelvis_height_adjustment_bound_m=.06,
                task_convention=task_convention,
                retarget_geometry_changed=True,source_timing_scale=1.,dynamics_qualified=False,
                hardware_authorized=False,source_provenance=source_provenance,
                reference_sha256=hashlib.sha256((output/"reference.npz").read_bytes()).hexdigest())
    (output/"report.json").write_text(json.dumps(report,indent=2))
    (output/"original_timeline.json").write_text(json.dumps(timeline,indent=2))
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--clip",choices=("pico","walk002","walk003","walk008"),required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--max-frames",type=int)
    a=p.parse_args()
    run(a.clip,a.output,a.max_frames)
