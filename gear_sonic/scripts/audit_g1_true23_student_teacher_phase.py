"""Read-only source-phase rescoring of preserved teacher physical trajectories."""
import argparse
import json
from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_true23_mpc_student import sha256


def run(args):
    source=json.loads((args.teacher/"report.json").read_text())
    timeline=json.loads(args.timeline.read_text())
    phase=next(p for p in timeline["phases"] if p["name"]=="source_motion")
    results=[]
    for case in source["case_summaries"]:
        trace_path=args.teacher/f"case_{case['case_id']:02d}.npz"
        assert sha256(trace_path)==case["trace_sha256"]
        with np.load(trace_path,allow_pickle=False) as archive:
            first=phase["control_start"];stop=min(case["completed_full_controls"],phase["control_stop"])
            idx=slice(first,stop)
            result=dict(case_id=case["case_id"],source_controls=max(0,stop-first),expert_eligible=case["expert_eligible"],
                        completed_full_controls=case["completed_full_controls"],failure=case["failure"],
                        range_excess_max=case["range_excess_max"],trace_sha256=case["trace_sha256"])
            if stop>first:
                result.update(source_root_p95_m=float(np.percentile(np.linalg.norm(archive["root_error"][idx],axis=1),95)),
                              source_leg_rmse_rad=float(np.sqrt(np.mean(archive["joint_error"][idx,:12]**2))),
                              source_arm_rmse_rad=float(np.sqrt(np.mean(archive["joint_error"][idx,13:]**2))),
                              source_feet_p95_m=np.percentile(archive["feet_error"][idx],95,axis=0).tolist(),
                              source_original_hand_head_p95_m=np.percentile(archive["original_task_error"][idx],95,axis=0).tolist())
            results.append(result)
    result=dict(kind="preserved_teacher_source_phase_rescoring",source_report_sha256=sha256(args.teacher/"report.json"),
                source_phase=phase,original_report_preserved=True,
                reason="initial collector metric slice included return/standing tail after source; eligibility/failure and all 3s metrics unaffected",
                cases=results)
    destination=args.teacher/"source_phase_metrics_v2.json"
    if destination.exists():raise FileExistsError(destination)
    destination.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(path=str(destination),nominal=results[0])),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--teacher",type=Path,required=True);p.add_argument("--timeline",type=Path,required=True)
    run(p.parse_args())
