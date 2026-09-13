"""Goal-conditioned native23 MPC student features and inference.

Explicit received-goal offsets cover38 frames (.74s). Clip identity, frame IDs,
clock time and planner states/gains never enter the network's feature vector.
"""
from pathlib import Path
import json
import hashlib

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

OFFSETS=np.array([0,1,2,4,8,16,24,37],dtype=np.int64)
FEATURES=1023
KIND="native23_mpc_actual_rollout_student_v1"


def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_inputs(bundle,clip="walk002"):
    bundle=Path(bundle); manifest=json.loads((bundle/"manifest.json").read_text())
    assert sha256(bundle/"native_prepared.xml")==manifest["portable_xml_sha256"]
    assert sha256(bundle/"prepared_model_arrays.npz")==manifest["prepared_arrays_sha256"]
    model=mujoco.MjModel.from_xml_path(str(bundle/"native_prepared.xml"))
    with np.load(bundle/"prepared_model_arrays.npz",allow_pickle=False) as data:
        for key in data.files: getattr(model,key)[:]=data[key]
        mujoco.mj_setConst(model,mujoco.MjData(model))
        for key in data.files: np.testing.assert_array_equal(getattr(model,key),data[key])
    contract=json.loads((bundle/"contract.json").read_text())
    values=[]
    for name in ("native_original.npz","original29.npz"):
        assert sha256(bundle/clip/name)==manifest["cases"][clip][name]
        with np.load(bundle/clip/name,allow_pickle=False) as data: values.append({k:data[k].copy() for k in data.files})
    timeline=json.loads((bundle/clip/"timeline.json").read_text())
    assert (model.nq,model.nv,model.nu)==(30,29,23) and model.opt.timestep==.002
    return model,contract,*values,timeline


class GoalFeatures:
    def __init__(self,motion,original,contract):
        self.motion,self.original,self.contract=motion,original,contract
        self.default=np.asarray(contract["default_q"])
        self.feet=[contract["body_names"].index(f"{s}_ankle_roll_link") for s in ("left","right")]
        self.root_rot=Rotation.from_quat(motion["body_quat_w"][:,0][:,[1,2,3,0]]).as_matrix()
        quats=original["source_task_quaternion_wxyz"]
        self.task_rot=Rotation.from_quat(quats.reshape(-1,4)[:,[1,2,3,0]]).as_matrix().reshape(len(quats),3,3,3)
        self.task_vel=np.diff(original["source_task_position_w"],axis=0,prepend=original["source_task_position_w"][:1])/.02
        self.task_omega=np.zeros_like(self.task_vel)
        for i in range(3):
            r=Rotation.from_matrix(self.task_rot[:,i])
            self.task_omega[1:,i]=(r[1:]*r[:-1].inv()).as_rotvec()/.02

    def __call__(self,qpos,qvel,previous_target,frame):
        frame=int(frame); slots=np.minimum(frame+OFFSETS,len(self.motion["joint_pos"])-1)
        actual_rot=Rotation.from_quat(np.asarray(qpos)[[4,5,6,3]]).as_matrix()
        yaw=np.arctan2(actual_rot[1,0],actual_rot[0,0]); c,s=np.cos(yaw),np.sin(yaw)
        heading=np.array([[c,-s,0],[s,c,0],[0,0,1]])
        root=np.asarray(qpos[:3]); mo=self.motion; orig=self.original
        proprio=np.r_[qpos[7:]-self.default,qvel[6:],qvel[3:6],actual_rot.T@np.array([0.,0.,-1.]),
                       np.asarray(previous_target)-self.default,np.asarray(qvel[:3])@heading,qpos[2]]
        # All world vectors become measured-heading coordinates. Rotation6D is
        # the first two columns, flattened consistently, without Euler angles.
        root_relative=(mo["body_pos_w"][slots,0]-root)@heading
        root_rotation=np.einsum("ij,njk->nik",heading.T,self.root_rot[slots])[:,:,:2].reshape(len(slots),6)
        feet_relative=(mo["body_pos_w"][slots][:,self.feet]-root)@heading
        task_relative=(orig["source_task_position_w"][slots]-root)@heading
        task_rotation=np.einsum("ij,ntjk->ntik",heading.T,self.task_rot[slots])[:,:,:,:2].reshape(len(slots),18)
        goal=np.concatenate((mo["joint_pos"][slots],mo["joint_vel"][slots],root_relative,root_rotation,
                             mo["body_lin_vel_w"][slots,0]@heading,mo["body_ang_vel_w"][slots,0]@heading,
                             feet_relative.reshape(len(slots),6),(mo["body_lin_vel_w"][slots][:,self.feet]@heading).reshape(len(slots),6),
                             task_relative.reshape(len(slots),9),task_rotation,
                             (self.task_vel[slots]@heading).reshape(len(slots),9),
                             (self.task_omega[slots]@heading).reshape(len(slots),9)),axis=1)
        assert proprio.shape==(79,) and goal.shape==(8,118)
        result=np.r_[proprio,goal.ravel()].astype(np.float32)
        if result.shape!=(FEATURES,) or not np.isfinite(result).all(): raise ValueError("invalid student features")
        return result


def make_actor():
    import torch
    actor=torch.nn.Sequential(torch.nn.Linear(FEATURES,256),torch.nn.ELU(),torch.nn.Linear(256,256),
                              torch.nn.ELU(),torch.nn.Linear(256,23))
    torch.nn.init.zeros_(actor[-1].weight); torch.nn.init.zeros_(actor[-1].bias)
    return actor


class StudentCPU:
    def __init__(self,path):
        import torch
        self.torch=torch
        self.saved=torch.load(Path(path),map_location="cpu",weights_only=True)
        if self.saved["kind"]!=KIND or self.saved["goal_offsets"]!=OFFSETS.tolist(): raise ValueError("student contract mismatch")
        self.actor=make_actor(); self.actor.load_state_dict(self.saved["actor_state"]);self.actor.eval()
        self.mean=self.saved["feature_mean"];self.std=self.saved["feature_std"]

    def predict(self,features,qref,limits):
        with self.torch.inference_mode():
            x=self.torch.from_numpy(features)[None]
            delta=self.actor((x-self.mean)/self.std)[0].numpy()*.5
        return np.clip(qref+delta,limits[:,0],limits[:,1])


def physical_failure(model,data,contract):
    limits=np.asarray(contract["joint_limits"])
    excess=float(max(0,np.max(limits[:,0]-data.qpos[7:]),np.max(data.qpos[7:]-limits[:,1])))
    speed=float(np.max(np.abs(data.qvel[6:])/contract["native_velocity"]))
    effort=float(np.max(np.abs(data.qfrc_actuator[6:])/contract["native_effort"]))
    tilt=float(np.arccos(np.clip(1-2*np.sum(data.qpos[4:6]**2),-1,1)))
    failure=None
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or data.qpos[2]<.25 or tilt>1.2:
        failure=dict(kind="fall_or_nonfinite",height=float(data.qpos[2]),tilt=tilt)
    elif excess>.01 or speed>1 or effort>1+1e-8:
        failure=dict(kind="physical_limit",range_excess=excess,velocity_ratio=speed,effort_ratio=effort)
    return failure,dict(range_excess=excess,velocity_ratio=speed,effort_ratio=effort)


def tracking_metrics(model,data,motion,original,contract,frame):
    mujoco.mj_kinematics(model,data)
    feet=[model.body(f"{s}_ankle_roll_link").id for s in ("left","right")]
    foot_src=np.asarray(feet)-1
    tasks=[]
    for body,local in [(model.body("left_wrist_roll_rubber_hand").id,(.264,-.025,0)),
                       (model.body("right_wrist_roll_rubber_hand").id,(.264,.025,0)),
                       (model.body("torso_link").id,(0,0,.35))]:
        tasks.append(data.xpos[body]+data.xmat[body].reshape(3,3)@np.asarray(local))
    root=data.qpos[:3]-motion["body_pos_w"][frame,0]
    return dict(joint_error=data.qpos[7:]-motion["joint_pos"][frame],root_error=root,
                feet_error=np.linalg.norm(data.xpos[feet]-motion["body_pos_w"][frame,foot_src],axis=1),
                original_task_error=np.linalg.norm(np.asarray(tasks)-original["source_task_position_w"][frame],axis=1),
                original_relative_task_error=np.linalg.norm((np.asarray(tasks)-data.qpos[:3])-(original["source_task_position_w"][frame]-original["source_qpos29"][frame,:3]),axis=1))
