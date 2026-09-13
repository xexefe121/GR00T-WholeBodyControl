"""Separate native self contacts that arm-only reference repair cannot move."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.utils.g1_true23_mpc_student import load_inputs,sha256


def run(args):
    results=[]
    for clip in args.clips:
        model,_,_,_,_=load_inputs(args.bundle,clip);data=mujoco.MjData(model)
        arm_roots={model.body(f"{side}_shoulder_pitch_link").id for side in ("left","right")}
        movable=set()
        for body in range(1,model.nbody):
            ancestor=body
            while ancestor:
                if ancestor in arm_roots:movable.add(body);break
                ancestor=int(model.body_parentid[ancestor])
        path=args.references/clip/"reference.npz"
        with np.load(path,allow_pickle=False) as a:motion={k:a[k].copy() for k in a.files}
        pairs={};maximum=np.zeros(len(motion["joint_pos"]))
        for frame in range(len(maximum)):
            data.qpos[:]=np.r_[motion["body_pos_w"][frame,0],motion["body_quat_w"][frame,0],motion["joint_pos"][frame]]
            data.qvel[:]=0.;mujoco.mj_forward(model,data)
            frame_pairs={}
            for contact in data.contact:
                ga,gb=int(contact.geom1),int(contact.geom2);ba,bb=int(model.geom_bodyid[ga]),int(model.geom_bodyid[gb])
                if contact.dist>=0 or not ba or not bb or ba in movable or bb in movable:continue
                key=f"{ga}:{gb}";depth=-float(contact.dist)
                frame_pairs[key]=max(frame_pairs.get(key,0.),depth)
                if key not in pairs:pairs[key]=dict(geom_ids=[ga,gb],bodies=[model.body(ba).name,model.body(bb).name],frames=[],depths_m=[])
                if depth>maximum[frame]:maximum[frame]=depth
            for key,depth in frame_pairs.items():pairs[key]["frames"].append(frame);pairs[key]["depths_m"].append(depth)
        compact=[]
        for value in pairs.values():
            index=int(np.argmax(value["depths_m"]))
            compact.append(dict(geom_ids=value["geom_ids"],bodies=value["bodies"],frames=len(value["frames"]),
                maximum_depth_m=value["depths_m"][index],worst_frame=value["frames"][index],
                frames_deeper_1mm=sum(d>.001 for d in value["depths_m"]),frames_deeper_5mm=sum(d>.005 for d in value["depths_m"])))
        results.append(dict(clip=clip,reference_sha256=sha256(path),all_frames=len(maximum),fixed_contact_frames=int(np.sum(maximum>0)),
                            fixed_contact_frames_deeper_1mm=int(np.sum(maximum>.001)),fixed_contact_frames_deeper_5mm=int(np.sum(maximum>.005)),
                            maximum_depth_m=float(maximum.max()),pairs=compact))
    report=dict(mujoco=mujoco.__version__,classification="both bodies have no arm joint in ancestor chain",
                actual_native_mj_forward_contacts=True,native_masks_exclusions_unchanged=True,results=results,
                note="Arm-only changes cannot alter these distances. Depth is reported, not automatically treated as a dynamic failure or permission to alter native collisions.")
    args.output.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--bundle",type=Path,required=True);p.add_argument("--references",type=Path,required=True)
    p.add_argument("--clips",type=lambda v:v.split(','),default=["pico","walk002","walk003","walk008"])
    p.add_argument("--output",type=Path,required=True);run(p.parse_args())
