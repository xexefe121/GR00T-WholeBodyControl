"""Compare identical completed control prefixes; keep whole-motion gates separate."""
import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
for path in ('/mnt/e/codex_sonic_runtime/bfm_seed_20260910/onnx_deps',
             '/root/.venvs/g1_true23_mjlab/lib/python3.11/site-packages'):
    sys.path.append(path)
from gear_sonic.scripts.evaluate_g1_true23_causal_dynamics import load_case,metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    model,contract,motion,original,timeline=load_case('pico')
    bank=Path('/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/causal_dynamics_v1/bank/bank.json')
    tasks=json.loads(bank.read_text())['tasks'];runs=[]
    for path in args.runs:
        report=json.loads((path/'report.json').read_text())
        if report['clip']!='pico' or report['standing_diagnostic']:raise ValueError('Pico motion runs required')
        with np.load(path/'trace.npz',allow_pickle=False) as z:states=z['states']
        runs.append((path,report,states))
    common=min((len(states)-1)//10 for _,_,states in runs)
    controls=np.arange(common);frames=np.minimum(controls+11,len(motion['joint_pos'])-1)
    output=[]
    for path,report,states in runs:
        prefix=metrics(model,states[(controls+1)*10,:30],frames,controls,motion,original,timeline,tasks)
        output.append(dict(run=str(path),matched_prefix=prefix,physical_complete=report['physical_complete'],
            whole_run_source=report['source'],physical_seconds=(len(states)-1)*.002,
            terminal_hold_reached=report.get('terminal_standing_reached',bool(report['physical_complete'])),
            terminal_hold_passed=bool(report['physical_complete'] and report['continuous_quiet30']['passed']),
            controller_deadline_misses=report['controller_deadline_misses'],
            late_physics_steps=report['plant_finishes_over2ms_late'],
            maximum_range_excess=report['maximum_range_excess']))
    result=dict(common_complete_controls=common,common_duration_seconds=common*.02,
        alignment='identical control IDs and reference frames, no root alignment',
        partial_prefix_is_not_whole_motion_acceptance=True,runs=output)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(str(args.output),flush=True)


if __name__=='__main__':main()
