"""Thirty-second standing regression for the same balanced controller API.

This isolated standing test does not qualify source motion or teleoperation.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import mujoco
from gear_sonic.scripts.evaluate_g1_true23_causal_dynamics import load_case,archive,assess,quiet,ROOT,BUNDLE,NEW
from gear_sonic.utils.g1_true23_causal_balance import BalancedCausalController
from gear_sonic.utils.g1_true23_bfmzero_stream import Packet


def main():
    p=argparse.ArgumentParser();p.add_argument('--actor',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--single-policy',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);results=[]
    model,c,motion,original,timeline=load_case('walk003')
    meta=json.loads((NEW/'causal_dynamics_v1/bank/bank.json').read_text())
    neutral={k:v[:1] for k,v in archive(BUNDLE/'walk003/native_original.npz').items() if k!='fps'}
    weights=ROOT/'artifacts/g1_true23_six_hour_replan_20260910_v1/bfmzero_inference_v1/inference.safetensors'
    fields={k:v[11].copy() for k,v in motion.items() if k!='fps'}
    for key in ('joint_vel','body_lin_vel_w','body_ang_vel_w'):fields[key][:]=0
    for velocity in (0.,.03,-.03):
        if a.single_policy:
            from gear_sonic.utils.g1_true23_causal_controller import CausalController
            controller=CausalController(a.actor,c,model=model,standing_qpos=timeline['configured_standing_qpos'],tasks=meta['tasks'])
        else:
            controller=BalancedCausalController(a.actor,c,weights=weights,neutral_motion=neutral,model=model,
                standing_qpos=timeline['configured_standing_qpos'],tasks=meta['tasks'])
        data=mujoco.MjData(model)
        data.qpos[:]=np.r_[motion['body_pos_w'][10,0],motion['body_quat_w'][10,0],motion['joint_pos'][10]]
        data.qvel[:]=0;data.qvel[0]=velocity;mujoco.mj_forward(model,data)
        qs=[data.qpos.copy()];vs=[data.qvel.copy()];targets=[];durations=[];failure=None
        for control in range(1500):
            now=control*.02
            assert controller.receive(Packet(0,control,now,fields),original['source_task_position_w'][11],original['source_task_quaternion_wxyz'][11],now)
            started=time.perf_counter();command=controller.command(data.qpos,data.qvel,now)
            durations.append((time.perf_counter()-started)*1000)
            if not a.single_policy:assert command.status['motion_fraction']==0
            target=command.targets;controller.commit_applied(data.qpos,data.qvel,target);targets.append(target)
            for sub in range(10):
                data.ctrl[:]=np.clip(c['kp']*(target-data.qpos[7:])-c['kd']*data.qvel[6:],-c['native_effort'],c['native_effort'])
                mujoco.mj_step(model,data);qs.append(data.qpos.copy());vs.append(data.qvel.copy())
                reasons,values=assess(data,c,(control*10+sub+1)*.002)
                if reasons:failure=dict(control=control,substep=sub+1,reasons=reasons,**values);break
            if failure:break
        complete=len(qs)==15001 and failure is None
        standing=quiet(np.asarray(qs),np.asarray(vs),original['source_qpos29'][11])
        report=dict(initial_root_x_velocity=velocity,physical_complete=complete,seconds=float(data.time),quiet=standing,
            failure=failure,passed=bool(complete and standing['passed']),policy_ms_p50_p95_max=np.percentile(durations,[50,95,100]).tolist(),
            full_body_motion_test=False,independent_realtime=False,hardware_authorized=False,single_policy=a.single_policy)
        results.append(report)
        np.savez_compressed(a.output/f'stand_{velocity:+.2f}.npz',qpos=qs,qvel=vs,target=targets)
        print(json.dumps(report),flush=True)
    (a.output/'report.json').write_text(json.dumps(dict(cases=results,standing_only_pass=all(r['passed'] for r in results),
        full_teleoperation_qualified=False),indent=2)+'\n')

if __name__=='__main__':main()
