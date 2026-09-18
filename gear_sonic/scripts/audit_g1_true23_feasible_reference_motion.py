"""Small adapter: audit standalone native23 motions with established geometry routines."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from gear_sonic.scripts.audit_g1_true23_fixed_self_contacts import measure_fixed_contacts
from gear_sonic.scripts.audit_g1_true23_reference_bank_self_contacts import (
    measure_self_contacts,
    validated_reference_qpos,
)
from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256, reference_geometry
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion",action="append",required=True,metavar="NAME=PATH")
    parser.add_argument("--asset-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.output.is_symlink(): raise FileExistsError("audit refuses overwrite")
    root=Path(__file__).resolve().parents[2]
    _,model,_=prepare_true23_model(args.asset_root/MODEL,root/PHYSICS)
    rows=[]
    for value in args.motion:
        name,text=value.split("=",1);path=Path(text).resolve(strict=True)
        with np.load(path,allow_pickle=False) as data: arrays={key:data[key].copy() for key in data.files}
        if "joint_names" not in arrays:
            # Imported and mjbatch motions omit joint names but store joint_pos in native23
            # hardware order. validated_reference_qpos rejects a wrong order independently:
            # its FK consistency check compares stored body poses with exact-model FK.
            arrays["joint_names"]=np.asarray(HARDWARE_23_JOINT_NAMES)
        poses,_=validated_reference_qpos(model,arrays)
        fixed=measure_fixed_contacts(model,poses)
        self_contacts=measure_self_contacts(model,poses)
        # The floor routine requires the exact library key set, without the joint names
        # supplied above for the self-contact validator.
        floor=reference_geometry(model,{key:value for key,value in arrays.items() if key!="joint_names"})
        rows.append(dict(name=name,path=str(path),sha256=sha256(path),frames=len(poses),fixed_contacts=fixed,
                         self_contacts=self_contacts,floor=floor))
        print(json.dumps(dict(name=name,frames=len(poses),fixed_frames=fixed["fixed_contact_frames"],
            fixed_depth_m=fixed["maximum_depth_m"],self_frames=self_contacts["frames_with_robot_robot_penetration"],
            self_depth_m=self_contacts["maximum_penetration_m"],floor_frames=floor["frames_with_floor_overlap"],
            floor_depth_m=floor["worst_floor_overlap_m"])),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(dict(kind="g1_true23_feasible_reference_motion_audit_v1",mujoco="3.2.3",
            physical_model=str((args.asset_root/MODEL).resolve()),physical_model_sha256=sha256(args.asset_root/MODEL),
            compiled_model_sha256=compiled_model_sha256(model),fixed_contact_classification="both bodies have no arm joint in ancestor chain",
            reused_tools=["audit_g1_true23_fixed_self_contacts.measure_fixed_contacts","audit_g1_true23_reference_bank_self_contacts.measure_self_contacts","g1_true23_reference_floor.reference_geometry"],
            results=rows),stream,indent=2,allow_nan=False)
        stream.write("\n")


if __name__=="__main__": main()
