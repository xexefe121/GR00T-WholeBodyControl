"""Pure 1000-feature state/received-goal builder. No BFM or history inputs."""
import numpy as np
from scipy.spatial.transform import Rotation
OFFSETS=np.array([0,1,2,4,8,16,24,37],dtype=np.int64)
FEATURES=1000

class DirectFeatures:
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

    def __call__(self,qpos,qvel,frame):
        frame=int(frame); slots=np.minimum(frame+OFFSETS,len(self.motion["joint_pos"])-1)
        actual_rot=Rotation.from_quat(np.asarray(qpos)[[4,5,6,3]]).as_matrix()
        yaw=np.arctan2(actual_rot[1,0],actual_rot[0,0]); c,s=np.cos(yaw),np.sin(yaw)
        heading=np.array([[c,-s,0],[s,c,0],[0,0,1]])
        root=np.asarray(qpos[:3]); mo=self.motion; orig=self.original
        proprio=np.r_[qpos[7:]-self.default,qvel[6:],qvel[3:6],actual_rot.T@np.array([0.,0.,-1.]),
                       np.asarray(qvel[:3])@heading,qpos[2]]
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
        assert proprio.shape==(56,) and goal.shape==(8,118)
        result=np.r_[proprio,goal.ravel()].astype(np.float32)
        if result.shape!=(FEATURES,) or not np.isfinite(result).all(): raise ValueError("invalid student features")
        return result
