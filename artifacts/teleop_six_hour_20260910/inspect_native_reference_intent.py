"""Measure original hand/head intent lost by the current native23 reference."""
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gear_sonic.scripts.evaluate_g1_true23_bfmzero import ROOT,DATA,MODEL,load_motion
from gear_sonic.utils.g1_true23_hand_frame_tasks import neutral_wrist_hand_tasks


def main():
    native=mujoco.MjModel.from_xml_path(str(ROOT.parent/"GR00T-WholeBodyControl"/MODEL))
    source=mujoco.MjModel.from_xml_path(str(ROOT.parent/"GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_29dof.xml"))
    tasks,convention=neutral_wrist_hand_tasks(source,native)
    tasks=[next(t for t in tasks if t.name==name) for name in ("left_hand","right_hand","head_proxy")]
    data=mujoco.MjData(native)
    rows=[]
    for clip in ("pico","walk002","walk003","walk008"):
        motion,timeline,_=load_motion(clip)
        phase=next(p for p in timeline["phases"] if p["name"]=="source_motion")
        frames=range(phase["control_start"]+11,phase["control_stop"]+11)
        path=DATA/("pico_freedancing_v1/optical_reference_v2/original29.npz" if clip=="pico" else
                   f"{clip}/original_source_bundle_v1/original_reference.npz")
        with np.load(path,allow_pickle=False) as z:
            wanted=z["source_task_position_w"][list(frames)]
            wanted_quat=z["source_task_quaternion_wxyz"][list(frames)]
        actual=[]; orientation=[]
        for frame in frames:
            data.qpos[:]=np.r_[motion["body_pos_w"][frame,0],motion["body_quat_w"][frame,0],motion["joint_pos"][frame]]
            mujoco.mj_kinematics(native,data)
            actual.append([data.xpos[native.body(t.target_body).id]+data.xmat[native.body(t.target_body).id].reshape(3,3)@t.target_point for t in tasks])
            orientation.append([data.xquat[native.body(t.target_body).id].copy() for t in tasks])
        delta=np.linalg.norm(np.asarray(actual)-wanted,axis=-1)
        angle=(Rotation.from_quat(np.asarray(orientation).reshape(-1,4)[:,[1,2,3,0]]).inv()*
               Rotation.from_quat(wanted_quat.reshape(-1,4)[:,[1,2,3,0]])).magnitude().reshape(-1,3)
        rows.append(dict(clip=clip,source_frames=len(frames),native_reference_original_task_p95_m=np.percentile(delta,95,axis=0).tolist(),
                         native_reference_original_orientation_p95_rad=np.percentile(angle,95,axis=0).tolist()))
    result=dict(cases=rows,convention=convention,
                interpretation="Error of current reference, not the minimum achievable by all native23 poses.",
                dynamics_executed=False,hardware_authorized=False)
    (ROOT/"artifacts/teleop_six_hour_20260910/native_reference_intent_error.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(rows),flush=True)


if __name__=="__main__":
    main()
