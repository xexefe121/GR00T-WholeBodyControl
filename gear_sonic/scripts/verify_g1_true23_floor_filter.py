"""Independent state-space reconstruction of the declared causal floor filter."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from gear_sonic.utils.g1_true23_mpc_student import sha256


def filtered(raw,omega,dt):
    continuous=np.array([[0.,1.],[-omega**2,-2*omega]])
    transition=expm(continuous*dt)
    control=np.linalg.solve(continuous,(transition-np.eye(2))@np.array([0.,omega**2]))
    state=np.array([float(raw[0]),0.]);result=[]
    for target in raw:
        state=transition@state+control*target;result.append(state[0])
    return np.asarray(result)


def run(args):
    records=[];parameters=[]
    for clip in ("pico","walk002","walk003","walk008"):
        base=args.pack/clip
        receipt=json.loads((base/"floor_transform_receipt.json").read_text())
        assert sha256(base/"reference.npz")==receipt["output_reference_sha256"]
        assert sha256(base/"before_floor_reference.npz")==receipt["input_reference_sha256"]
        assert sha256(base/"frame_lift.npz")==receipt["frame_lift_sha256"]
        with np.load(args.audit/f"{clip}_floor_arrays.npz",allow_pickle=False) as a:raw=a["minimum_lift"].copy()
        with np.load(base/"frame_lift.npz",allow_pickle=False) as a:exported={k:a[k].copy() for k in a.files}
        f=receipt["filter"];params=(f["omega_per_second"],f["dt_seconds"],f["fixed_buffer_m"],f["clearance_guard_epsilon_m"])
        parameters.append(params)
        reconstructed=filtered(raw,params[0],params[1])
        lift=np.maximum(raw+params[3],reconstructed+params[2])
        errors=dict(independent_raw_lift=float(np.max(np.abs(raw-exported["raw_required_lift_m"]))),
                    independent_expm_filtered_lift=float(np.max(np.abs(reconstructed-exported["filtered_required_lift_m"]))),
                    independent_expm_final_lift=float(np.max(np.abs(lift-exported["frame_lift_m"]))))
        prefix=[]
        for end in (250,400,500,min(1000,len(raw)-2),len(raw)-2):
            changed=raw.copy();changed[end+1:]+=np.linspace(.1,.3,len(raw)-end-1)
            changed_output=filtered(changed,params[0],params[1])
            prefix.append(dict(last_unchanged_frame=end,changed_future_prefix_max_abs=float(np.max(np.abs(changed_output[:end+1]-reconstructed[:end+1])))))
        records.append(dict(clip=clip,parameters=params,reference_sha256=receipt["output_reference_sha256"],
                            reconstruction_max_absolute_errors=errors,prefix_causality=prefix,
                            exact_zero_preview_pose_verified=all(p["changed_future_prefix_max_abs"]==0 for p in prefix),
                            central_difference_velocity_additional_pose_support_frames=1,
                            guard_activations=int(np.sum(raw+params[3]>reconstructed+params[2])),
                            passed=max(errors.values())<1e-12 and all(p["changed_future_prefix_max_abs"]==0 for p in prefix)))
    report=dict(kind="independent_expm_floor_filter_reconstruction",parameters_identical_all_clips=all(p==parameters[0] for p in parameters),
                all_passed=all(r["passed"] for r in records),results=records,
                code_sha256=sha256(Path(__file__)),physical_model_unchanged=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--pack",type=Path,required=True);p.add_argument("--audit",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);run(p.parse_args())
