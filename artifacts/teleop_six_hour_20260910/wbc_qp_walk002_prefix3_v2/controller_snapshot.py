"""Native23 inverse-dynamics feasibility prototype, executed through joint torques.

Contact forces exist only at measured foot/ground collision points. Source foot
height and speed choose desired stance; measured contacts remain unilateral.
No simulator state, contact, external force, or physical parameter is changed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import clarabel
import mujoco
import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bundle(directory, clip):
    assert mujoco.__version__ == "3.2.3", "This prototype's verified Jacobian convention targets MuJoCo3.2.3"
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in (("native_prepared.xml", manifest["portable_xml_sha256"]),
                           ("prepared_model_arrays.npz", manifest["prepared_arrays_sha256"])):
        assert digest(directory/name) == expected
    model = mujoco.MjModel.from_xml_path(str(directory / "native_prepared.xml"))
    with np.load(directory / "prepared_model_arrays.npz", allow_pickle=False) as archive:
        for name in archive.files:
            getattr(model, name)[:] = archive[name]
        mujoco.mj_setConst(model, mujoco.MjData(model))
        for name in archive.files:
            np.testing.assert_array_equal(getattr(model, name), archive[name])
    contract = json.loads((directory / "contract.json").read_text())
    assert (model.nq, model.nv, model.nu) == (30, 29, 23)
    assert model.opt.timestep == .002
    arrays = []
    for name in ("native_original.npz", "original29.npz"):
        path = directory / clip / name
        assert digest(path) == manifest["cases"][clip][name]
        with np.load(path, allow_pickle=False) as archive:
            arrays.append({k:archive[k].copy() for k in archive.files})
    timeline = json.loads((directory / clip / "timeline.json").read_text())
    return model, contract, *arrays, timeline, manifest


def rotation_error(target_wxyz, actual_matrix):
    desired = Rotation.from_quat(np.asarray(target_wxyz)[[1,2,3,0]]).as_matrix()
    return Rotation.from_matrix(desired @ actual_matrix.T).as_rotvec()


@dataclass
class QPConfig:
    position_kp: float = 100.
    position_kd: float = 20.
    rotation_kp: float = 100.
    rotation_kd: float = 20.
    root_weight: float = 10.
    root_rotation_weight: float = 20.
    foot_weight: float = 30.
    hand_weight: float = 1.
    head_weight: float = 2.
    posture_weight: float = .05
    contact_weight: float = 100.
    friction_coefficient_cap: float = .6
    stance_height_above_initial_m: float = .025
    stance_speed_max_mps: float = .35
    joint_limit_barrier_frequency: float = 20.
    joint_limit_margin: float = .005
    acceleration_max: float = 500.
    task_acceleration_max: float = 50.


class Native23WBCQP:
    def __init__(self, model, contract, motion, original, config=None):
        self.model, self.contract = model, contract
        self.motion, self.original = motion, original
        self.config = config or QPConfig()
        self.foot_ids = [model.body(f"{s}_ankle_roll_link").id for s in ("left", "right")]
        self.hand_ids = [model.body(f"{s}_wrist_roll_rubber_hand").id for s in ("left", "right")]
        self.root_id, self.torso_id = model.body("pelvis").id, model.body("torso_link").id
        self.effort, self.velocity = [np.asarray(contract[k]) for k in ("native_effort", "native_velocity")]
        self.lo, self.hi = np.asarray(contract["joint_limits"]).T
        self.foot_initial_z = motion["body_pos_w"][10, np.asarray(self.foot_ids)-1, 2]
        self.task_velocity = np.diff(original["source_task_position_w"], axis=0,
                                     prepend=original["source_task_position_w"][:1]) / .02
        self.task_acceleration = np.diff(self.task_velocity, axis=0, prepend=self.task_velocity[:1]) / .02
        self.task_angular_velocity = np.zeros_like(self.task_velocity)
        quaternions=original["source_task_quaternion_wxyz"]
        for task_id in range(3):
            rotations=Rotation.from_quat(quaternions[:,task_id][:,[1,2,3,0]])
            self.task_angular_velocity[1:,task_id]=(rotations[1:]*rotations[:-1].inv()).as_rotvec()/.02
        self.task_angular_acceleration=np.diff(self.task_angular_velocity,axis=0,prepend=self.task_angular_velocity[:1])/.02
        self.body_acceleration = np.diff(motion["body_lin_vel_w"], axis=0,
                                         prepend=motion["body_lin_vel_w"][:1]) / .02
        self.body_angular_acceleration = np.diff(motion["body_ang_vel_w"], axis=0,
                                                 prepend=motion["body_ang_vel_w"][:1]) / .02
        self.joint_acceleration = np.diff(motion["joint_vel"], axis=0, prepend=motion["joint_vel"][:1]) / .02
        self.total_mass = float(model.body_mass.sum())
        self.settings = clarabel.DefaultSettings()
        self.settings.verbose = False
        self.settings.max_iter = 60
        self.settings.tol_gap_abs = 1e-7
        self.settings.tol_feas = 1e-7

    def point(self, data, body, local=None):
        point = data.xpos[body].copy()
        if local is not None:
            point += data.xmat[body].reshape(3,3) @ np.asarray(local)
        jac = np.zeros((3,29)); rot = np.zeros((3,29))
        dot = np.zeros((3,29)); rot_dot = np.zeros((3,29))
        mujoco.mj_jac(self.model, data, jac, rot, point, body)
        mujoco.mj_jacDot(self.model, data, dot, rot_dot, point, body)
        # In the pinned3.2.3 API, translation returned by mj_jacDot contracts
        # to spatial acceleration. Convert to classical point acceleration.
        # Independent moving-point finite differences verify this correction.
        bias=dot@data.qvel+np.cross(rot@data.qvel,jac@data.qvel)
        return point, jac, rot, bias, rot_dot @ data.qvel

    def contacts(self, data, frame):
        desired = []
        for i, body in enumerate(self.foot_ids):
            pos = self.motion["body_pos_w"][frame, body-1]
            speed = np.linalg.norm(self.motion["body_lin_vel_w"][frame, body-1])
            desired.append(bool(pos[2] <= self.foot_initial_z[i] + self.config.stance_height_above_initial_m
                                and speed <= self.config.stance_speed_max_mps))
        contacts=[]
        for i in range(data.ncon):
            c=data.contact[i]
            b1,b2=self.model.geom_bodyid[c.geom1],self.model.geom_bodyid[c.geom2]
            foot = b2 if b1 == 0 else b1 if b2 == 0 else -1
            if foot not in self.foot_ids or c.dist > 0 or c.exclude != 0:
                continue
            normal = np.asarray(c.frame[:3]) * (1 if b1 == 0 else -1)
            if normal[2] < .99:
                continue
            j=np.zeros((3,29)); jr=np.zeros((3,29)); jd=np.zeros((3,29))
            mujoco.mj_jac(self.model,data,j,jr,c.pos,foot)
            mujoco.mj_jacDot(self.model,data,jd,None,c.pos,foot)
            contacts.append(dict(index=i,foot=int(foot),pos=np.asarray(c.pos).copy(),jac=j,
                                 jdot_v=jd @ data.qvel+np.cross(jr@data.qvel,j@data.qvel),dist=float(c.dist),
                                 friction=min(float(c.friction[0]),self.config.friction_coefficient_cap),
                                 desired_stance=desired[self.foot_ids.index(foot)]))
        return contacts, desired

    def solve(self, data, frame):
        started=time.perf_counter()
        cfg=self.config
        contacts,desired=self.contacts(data,frame)
        nc=len(contacts); n=29+3*nc+23; tau_start=29+3*nc
        mass=np.empty((29,29)); mujoco.mj_fullM(self.model,mass,data.qM)
        eq=np.zeros((29,n)); eq[:,:29]=mass
        eq[:,tau_start:]=-np.r_[np.zeros((6,23)),np.eye(23)]
        for i,c in enumerate(contacts): eq[:,29+3*i:32+3*i]=-c["jac"].T
        # Preserve native free-root frictionloss: use its currently measured
        # constraint force as a known disturbance, never an applied root input.
        assert not mujoco.mj_isSparse(self.model)
        friction_rows=data.efc_type==mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF
        friction_force=data.efc_J.reshape(data.nefc,29)[friction_rows].T@data.efc_force[friction_rows]
        eq_rhs=data.qfrc_passive-data.qfrc_bias+friction_force
        tasks=[]; goals=[]

        def task(matrix, goal, weight):
            row=np.zeros((len(goal),n)); row[:,:29]=matrix
            tasks.append(np.sqrt(weight)*row); goals.append(np.sqrt(weight)*goal)

        def body_task(body,pos,quat,vel,ang_vel,acc,ang_acc,weight,rotweight,local=None):
            point,jac,rot,jdotv,rdotv=self.point(data,body,local)
            ades=acc+cfg.position_kp*(pos-point)+cfg.position_kd*(vel-jac@data.qvel)
            task(jac,np.clip(ades,-cfg.task_acceleration_max,cfg.task_acceleration_max)-jdotv,weight)
            if rotweight:
                rerr=rotation_error(quat,data.xmat[body].reshape(3,3))
                ades=ang_acc+cfg.rotation_kp*rerr+cfg.rotation_kd*(ang_vel-rot@data.qvel)
                task(rot,np.clip(ades,-cfg.task_acceleration_max,cfg.task_acceleration_max)-rdotv,rotweight)

        for body,weight,rweight in [(self.root_id,cfg.root_weight,cfg.root_rotation_weight),
                                    (self.foot_ids[0],cfg.foot_weight,3.),
                                    (self.foot_ids[1],cfg.foot_weight,3.)]:
            i=body-1
            body_task(body,self.motion["body_pos_w"][frame,i],self.motion["body_quat_w"][frame,i],
                      self.motion["body_lin_vel_w"][frame,i],self.motion["body_ang_vel_w"][frame,i],
                      self.body_acceleration[frame,i],self.body_angular_acceleration[frame,i],weight,rweight)
        for i,(body,local,weight,rweight) in enumerate([
            (self.hand_ids[0],(.264,-.025,0),cfg.hand_weight,.1),
            (self.hand_ids[1],(.264,.025,0),cfg.hand_weight,.1),
            (self.torso_id,(0,0,.35),cfg.head_weight,5.)]):
            body_task(body,self.original["source_task_position_w"][frame,i],
                      self.original["source_task_quaternion_wxyz"][frame,i],self.task_velocity[frame,i],
                      self.task_angular_velocity[frame,i],self.task_acceleration[frame,i],
                      self.task_angular_acceleration[frame,i],weight,rweight,local)
        posture=self.joint_acceleration[frame]+50*(self.motion["joint_pos"][frame]-data.qpos[7:])+10*(self.motion["joint_vel"][frame]-data.qvel[6:])
        task(np.c_[np.zeros((23,6)),np.eye(23)],posture,cfg.posture_weight)
        inequalities=[]; bounds=[]
        for i,c in enumerate(contacts):
            vel=c["jac"]@data.qvel
            desired_acc=-40*vel
            desired_acc[2] += -400*c["dist"]
            # Contact acceleration is soft; no unsupported foot is welded.
            task(c["jac"],desired_acc-c["jdot_v"],cfg.contact_weight if c["desired_stance"] else cfg.contact_weight*.1)
            offset=29+3*i; mu=c["friction"]/np.sqrt(2.)
            local=np.array([[1,0,-mu],[-1,0,-mu],[0,1,-mu],[0,-1,-mu],[0,0,-1],[0,0,1]])
            row=np.zeros((6,n)); row[:,offset:offset+3]=local
            inequalities.append(row); bounds.append(np.array([0,0,0,0,0,self.total_mass*9.81*2]))
            # Prevent a commanded inward normal acceleration at active contacts.
            row=np.zeros((1,n)); row[0,:29]=-c["jac"][2]
            inequalities.append(row); bounds.append(np.array([c["jdot_v"][2]-desired_acc[2]]))

        # Native effort, next-step speed/position, and early joint-limit braking.
        row=np.zeros((46,n)); row[:23,tau_start:]=np.eye(23); row[23:,tau_start:]=-np.eye(23)
        inequalities.append(row); bounds.append(np.r_[self.effort,self.effort])
        q,v=data.qpos[7:],data.qvel[6:]; dt=self.model.opt.timestep; w=cfg.joint_limit_barrier_frequency
        upper=np.minimum.reduce([np.full(23,cfg.acceleration_max),(self.velocity-v)/dt,
                                  (self.hi-q-dt*v)/dt**2,w*w*(self.hi-cfg.joint_limit_margin-q)-2*w*v])
        lower=np.maximum.reduce([np.full(23,-cfg.acceleration_max),(-self.velocity-v)/dt,
                                  (self.lo-q-dt*v)/dt**2,w*w*(self.lo+cfg.joint_limit_margin-q)-2*w*v])
        row=np.zeros((46,n)); row[:23,6:29]=np.eye(23); row[23:,6:29]=-np.eye(23)
        inequalities.append(row); bounds.append(np.r_[upper,-lower])
        stack=np.vstack(tasks); goal=np.concatenate(goals)
        reg=np.r_[np.full(29,1e-5),np.full(3*nc,1e-6),1e-4/self.effort**2]
        P=2*(stack.T@stack+np.diag(reg)); linear=-2*stack.T@goal
        inequality=np.vstack(inequalities); rhs=np.concatenate(bounds)
        A=np.vstack((eq,inequality)); b=np.r_[eq_rhs,rhs]
        solver=clarabel.DefaultSolver(sparse.csc_matrix(np.triu(P)),linear,sparse.csc_matrix(A),b,
                                      [clarabel.ZeroConeT(29),clarabel.NonnegativeConeT(len(rhs))],self.settings)
        solution=solver.solve(); status=str(solution.status)
        x=np.asarray(solution.x)
        solved=status in ("Solved","AlmostSolved") and np.isfinite(x).all()
        report=dict(status=status,solved=bool(solved),contacts=nc,desired_stance=desired,
                    measured_contact_points=[dict(foot=c["foot"],pos=c["pos"].tolist(),dist=c["dist"],
                                                   desired_stance=c["desired_stance"]) for c in contacts],
                    solver_ms=float(solution.solve_time*1000),total_ms=(time.perf_counter()-started)*1000,
                    iterations=solution.iterations,unilateral_acceleration_rows=nc,
                    dynamics_equality_max=float(np.max(np.abs(eq@x-eq_rhs))) if solved else None,
                    inequality_excess_max=float(max(0,np.max(inequality@x-rhs))) if solved else None,
                    task_rms=float(np.sqrt(np.mean((stack@x-goal)**2))) if solved else None,
                    acceleration_bound_conflict=float(np.max(lower-upper)),
                    measured_dof_friction_force=friction_force.tolist(),
                    config=asdict(cfg))
        return (x[tau_start:].copy() if solved else None),report,(
            dict(qacc=x[:29].copy(),contact_forces=x[29:tau_start].reshape(-1,3).copy()) if solved else {})
