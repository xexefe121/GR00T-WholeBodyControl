"""Native/Python prediction parity on archived states, all 49 delay pairs."""
import argparse,json
from pathlib import Path
import mujoco
import numpy as np
from diagnose_pico_preview_failure import load_case,set_pd
from gear_sonic.scripts.evaluate_g1_true23_causal_dynamics import assess
from gear_sonic.utils.g1_true23_sustained_braking import Predictor


def python_row(m,c,q,v,previous,first,nominal,goal,j,a,b):
    d=mujoco.MjData(m);d.qpos[:]=q;d.qvel[:]=v;mujoco.mj_forward(m,d)
    row=np.zeros(11);row[3]=1e9;row[10]=-1;last=previous.copy();target=first.copy();streak=0
    for step in range(40):
        cycle,sub=divmod(step,10);delay=b if cycle else a
        if step and not sub:
            last=target.copy();target=nominal.copy()
            mujoco.mj_forward(m,d);unit=np.zeros((1,29));unit[0,6+j]=1;inverse=np.zeros_like(unit)
            mujoco.mj_solveM(m,d,inverse,unit)
            inertia=1/max(inverse[0,6+j],1e-9)
            accel=400*(goal-d.qpos[7+j])-40*d.qvel[6+j]
            torque=np.clip(inertia*accel+d.qfrc_bias[6+j]-d.qfrc_passive[6+j],-c['native_effort'][j],c['native_effort'][j])
            target[j]=np.clip(d.qpos[7+j]+(torque+c['kd'][j]*d.qvel[6+j])/c['kp'][j],*c['joint_limits'][j])
        raw=set_pd(d,last if sub<delay else target,c['kp'],c['kd'],c['native_effort'],c['joint_limits'])
        row[6]=max(row[6],np.max(abs(raw)/c['native_effort']))
        mujoco.mj_step(m,d);reasons,_=assess(d,c,(step+1)*.002)
        margin=np.minimum(d.qpos[7:]-c['joint_limits'][:,0],c['joint_limits'][:,1]-d.qpos[7:])
        if margin.min()<row[3]:row[3]=margin.min();row[10]=margin.argmin()
        row[4]=max(row[4],np.max(abs(d.qvel[6:])/c['native_velocity']))
        row[5]=max(row[5],np.max(abs(d.qfrc_actuator[6:])/c['native_effort']))
        reason=(1 if margin.min()<.0001 else 0)|(2 if row[4]>1 else 0)|(4 if row[5]>1+1e-9 else 0)
        reason|=8 if 'fall' in reasons else 0;reason|=16 if 'nonfinite' in reasons else 0
        reason|=32 if 'engine_warning' in reasons else 0;reason|=64 if any(x in reasons for x in ('quaternion','clock')) else 0
        if reason:row[1]=step+1;row[2]=reason;break
        if sub==9 and cycle>=2:streak=streak+1 if margin[j]>=.02 and abs(d.qvel[6+j])<=.25 else 0
    row[7]=d.qpos[7+j];row[8]=d.qvel[6+j];row[9]=streak>=2
    if not row[2] and not row[9]:row[1]=40;row[2]=128
    row[0]=not row[2]
    return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();m,c,*_=load_case('pico');predictor=Predictor(m,c,args.library)
    base=Path('/mnt/e/codex-artifacts/native23_shoulder_recovery_20260914/replay_v1')
    run=Path('/mnt/e/codex-artifacts/native23_preview_repair_20260913/strict_newer_v1')
    diagnostics={r['control']:r for r in map(json.loads,(run/'preview_diagnostics.jsonl').read_text().splitlines())}
    cases=[]
    for control in (3159,3168,3169):
        with np.load(base/f'boundary_{control}.npz') as z:q=z['qpos'];v=z['qvel'];previous=z['previous_target']
        nominal=np.array(diagnostics[control]['requested_target']);j=predictor.joint
        goal=float(np.clip(q[7+j],c['joint_limits'][j,0]+.15,c['joint_limits'][j,1]-.15))
        actual=predictor.evaluate(q,v,previous,nominal,nominal,goal)['rows']
        expected=np.array([python_row(m,c,q,v,previous,nominal,nominal,goal,j,a,b) for a in range(7) for b in range(7)])
        np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-10)
        np.testing.assert_array_equal(actual[:,[0,1,2,9,10]],expected[:,[0,1,2,9,10]])
        cases.append(dict(control=control,delay_pairs=49,passing_pairs=int(actual[:,0].sum()),maximum_error=float(abs(actual-expected).max())))
    args.output.write_text(json.dumps(dict(passed=True,cases=cases,mujoco_version=mujoco.__version__),indent=2)+'\n')
    print(args.output.read_text())


if __name__=='__main__':main()
