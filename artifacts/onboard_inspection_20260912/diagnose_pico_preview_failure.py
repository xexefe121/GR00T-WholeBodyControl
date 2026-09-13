"""Replay actual archived commands; qualify solver continuation before using it.

No actor request or packet-admission history is invented from applied targets.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
for path in ('/mnt/e/codex_sonic_runtime/bfm_seed_20260910/onnx_deps',
             '/root/.venvs/g1_true23_mjlab/lib/python3.11/site-packages'):
    sys.path.append(path)
from gear_sonic.scripts.evaluate_g1_true23_causal_dynamics import load_case
from gear_sonic.utils.g1_true23_native_preview_guard import NativePreviewGuard
from gear_sonic.utils.g1_true23_sim_preview import json_safe


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(exist_ok=False)
    fw=Path('/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/onboard_factory_firmware_v1')
    run=fw/'received_sim_runs/20260913_111758_661_pico'
    report=json.loads((run/'report.json').read_text())
    with np.load(run/'trace.npz') as z:
        states=z['states'];targets=z['targets'];torques=z['torques'];timing=z['timing']
    model,contract,*_=load_case('pico')
    data=mujoco.MjData(model);data.qpos[:]=states[0,:30];data.qvel[:]=states[0,30:59]
    mujoco.mj_forward(model,data)
    kp=np.asarray(report['actual_pd_kp']);kd=np.asarray(report['actual_pd_kd']);effort=contract['native_effort']
    error=np.zeros((len(targets),3));saved={}
    for step,target in enumerate(targets):
        if step in (31690,31810,31820):saved[step//10]=copy.copy(data)
        data.ctrl[:]=np.clip(kp*(target-data.qpos[7:])-kd*data.qvel[6:],-effort,effort)
        error[step,2]=np.max(np.abs(data.ctrl-torques[step]))
        mujoco.mj_step(model,data)
        error[step,:2]=[np.max(np.abs(data.qpos-states[step+1,:30])),
                       np.max(np.abs(data.qvel-states[step+1,30:59]))]
    exact=bool(np.all(error==0))
    np.save(args.output/'every_2ms_replay_error.npy',error)
    guard=NativePreviewGuard(model,contract,delay_substeps=6,
                             library=fw/'native_preview_backend_v2/libtrue23preview.so')
    rows=[]
    for control in (3169,3181,3182):
        boundary=control*10
        selected=np.flatnonzero(timing[:,4]==control)
        first=int(selected[0]);target=targets[first].copy();previous=targets[boundary-1].copy()
        # Cold reset, explicitly distinct from the validated replay data above.
        guard.native.reset(states[boundary,:30],states[boundary,30:59],previous)
        low,high=guard.native.preview(target)
        joint=int(np.argmax(np.maximum(low,high)))
        cold=dict(lower_excess_rad=low,upper_excess_rad=high,
                  maximum_excess_rad=float(max(low.max(),high.max())),
                  solver_initialization='fresh MjData + mj_forward; cold-solver diagnostic only')
        continuations=[]
        if exact:
            for delay in (0,6,first-boundary):
                continuing=copy.copy(saved[control]);low_live=np.full(23,-np.inf);high_live=low_live.copy()
                for substep in range(16):
                    applied=previous if substep<delay else target
                    continuing.ctrl[:]=np.clip(kp*(applied-continuing.qpos[7:])-kd*continuing.qvel[6:],-effort,effort)
                    mujoco.mj_step(model,continuing)
                    low_live=np.maximum(low_live,contract['joint_limits'][:,0]+1e-4-continuing.qpos[7:])
                    high_live=np.maximum(high_live,continuing.qpos[7:]-contract['joint_limits'][:,1]+1e-4)
                continuations.append(dict(old_target_substeps=delay,
                    lower_excess_rad=low_live,upper_excess_rad=high_live,
                    maximum_excess_rad=float(max(low_live.max(),high_live.max()))))
        rows.append(dict(control=control,simulation_boundary_seconds=boundary*.002,
            historical_applied_candidate=target,previous_applied_target=previous,
            raw_actor_request_available=False,first_application_step=first,old_target_substeps=first-boundary,
            scheduled_application_delay_ms=(first-boundary+1)*2,
            limiting_joint_index=joint,limiting_joint_name=model.joint(joint+1).name,
            cold_preview=cold,validated_solver_continuations=continuations))
    result=dict(archived_run=str(run),physics_steps_checked=len(targets),
        initial_state='archived FP64 initial q/v; fresh MjData; mj_forward',
        every_2ms_max_errors=dict(zip(('qpos','qvel','commanded_torque'),error.max(axis=0).tolist())),
        exact_applied_target_replay=exact,controller_continuation_reconstructed=False,
        hypothetical_commands_held_for_full_32ms=True,cases=rows)
    (args.output/'report.json').write_text(json.dumps(json_safe(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)


if __name__=='__main__':main()
