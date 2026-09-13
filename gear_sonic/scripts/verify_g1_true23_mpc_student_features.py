"""Check spatial invariance and the explicit38-frame feature access boundary."""
import copy
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from gear_sonic.utils.g1_true23_mpc_student import GoalFeatures,load_inputs,OFFSETS

bundle=Path("artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1")
_,contract,motion,original,_=load_inputs(bundle)
frame=400
qpos=np.r_[motion["body_pos_w"][frame-1,0],motion["body_quat_w"][frame-1,0],motion["joint_pos"][frame-1]]
qvel=np.r_[motion["body_lin_vel_w"][frame-1,0],[.1,-.2,.3],motion["joint_vel"][frame-1]]
previous=motion["joint_pos"][frame-1]+.02
x=GoalFeatures(motion,original,contract)(qpos,qvel,previous,frame)
rot=Rotation.from_euler("z",.72);translation=np.array([1.3,-.7,0.])
mo=copy.deepcopy(motion);orig=copy.deepcopy(original)
for key in ("body_pos_w","body_lin_vel_w","body_ang_vel_w"):
    mo[key]=rot.apply(mo[key].reshape(-1,3)).reshape(mo[key].shape)
mo["body_pos_w"]+=translation
rq=rot*Rotation.from_quat(mo["body_quat_w"].reshape(-1,4)[:,[1,2,3,0]])
mo["body_quat_w"]=rq.as_quat()[:,[3,0,1,2]].reshape(mo["body_quat_w"].shape)
orig["source_task_position_w"]=rot.apply(orig["source_task_position_w"].reshape(-1,3)).reshape(orig["source_task_position_w"].shape)+translation
rq=rot*Rotation.from_quat(orig["source_task_quaternion_wxyz"].reshape(-1,4)[:,[1,2,3,0]])
orig["source_task_quaternion_wxyz"]=rq.as_quat()[:,[3,0,1,2]].reshape(orig["source_task_quaternion_wxyz"].shape)
orig["source_qpos29"][:,:3]=rot.apply(orig["source_qpos29"][:,:3])+translation
q2=qpos.copy();v2=qvel.copy();q2[:3]=rot.apply(qpos[:3])+translation
q2[3:7]=(rot*Rotation.from_quat(qpos[[4,5,6,3]])).as_quat()[[3,0,1,2]]
v2[:3]=rot.apply(qvel[:3])
transformed=GoalFeatures(mo,orig,contract)(q2,v2,previous,frame)
invariance_error=float(np.max(np.abs(x-transformed)))
mo=copy.deepcopy(motion);orig=copy.deepcopy(original)
boundary=frame+int(OFFSETS[-1])+1
for key in ("joint_pos","joint_vel","body_pos_w","body_lin_vel_w","body_ang_vel_w"):
    mo[key][boundary:]+=3.
orig["source_task_position_w"][boundary:]+=5.
future=GoalFeatures(mo,orig,contract)(qpos,qvel,previous,frame)
future_error=float(np.max(np.abs(x-future)))
result=dict(passed=invariance_error<1e-5 and future_error==0,feature_size=len(x),
            horizontal_translation_and_yaw_invariance_max_error=invariance_error,
            changes_after_frame_plus37_have_max_error=future_error,
            velocity_contract="uses supplied native goal velocity channels; original task velocities derive backward from received current/past poses",
            goal_offsets=OFFSETS.tolist(),received_goal_frames=38)
out=Path("E:/codex_sonic_runtime/mpc_student_20260910/feature_verification_v1.json")
out.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result));assert result["passed"]
