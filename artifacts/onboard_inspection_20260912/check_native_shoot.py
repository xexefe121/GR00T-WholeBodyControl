"""Native lookahead must reproduce ordinary PD physics for a held target."""
from pathlib import Path
import sys
import json
import time
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from artifacts.teleop_resume_20260911.run_causal_native_clock import NEW
from gear_sonic.scripts.evaluate_g1_true23_causal_dynamics import load_case,archive
from gear_sonic.utils.g1_true23_native_shoot import NativePoseLookahead,ptr
from gear_sonic.utils.g1_true23_received_features import prepare_reference
from gear_sonic.utils.g1_true23_factory_locomotion import IDS
import yaml

fw=NEW/'onboard_factory_firmware_v1';model,c,motion,original,timeline=load_case('walk002')
meta=json.loads((NEW/'causal_dynamics_v1/bank/bank.json').read_text())
cfg=yaml.safe_load((fw/'decoded_configs/policies/human_loco/fsm_human_loco_config.yaml').read_text(encoding='utf-8'))
kp=np.asarray(cfg['joint_kp'])[IDS];kd=np.asarray(cfg['joint_kd'])[IDS]
initial=archive(NEW/'causal_dynamics_v1/bank/walk002.npz')['states'][10]
look=NativePoseLookahead(fw/'native_shoot_v1/libtrue23shoot.so',model,c,kp,kd,meta['tasks'])
reference=prepare_reference({k:v[:12] for k,v in motion.items() if k!='fps'},
    {k:v[:12] for k,v in original.items() if k.startswith('source_task_')},c)
target=np.ascontiguousarray(initial[7:30][None],np.float64);q=np.ascontiguousarray(initial[:30]);v=np.ascontiguousarray(initial[30:])
goal=np.zeros(125);goal[52]=1;score=np.zeros(1);terminal=np.zeros((1,59))
index=look.lib.shoot_evaluate(look.context,ptr(q),ptr(v),ptr(target),1,30,ptr(goal),ptr(score),ptr(terminal))
assert index==0 and score[0]<1e9
d=mujoco.MjData(model);d.qpos[:]=q;d.qvel[:]=v;mujoco.mj_forward(model,d)
for step in range(30):
    band=np.minimum(.1,np.diff(c['joint_limits'],axis=1).ravel()*.2)
    penetration=d.qpos[7:]-np.clip(d.qpos[7:],c['joint_limits'][:,0]+band,c['joint_limits'][:,1]-band)
    outward=np.where(penetration*d.qvel[6:]>0,d.qvel[6:],0)
    bounded=np.clip(target[0],c['joint_limits'][:,0]+.06,c['joint_limits'][:,1]-.06)
    d.ctrl[:]=np.clip(kp*(bounded-d.qpos[7:])-kd*d.qvel[6:]-100*penetration-2*outward,-c['native_effort'],c['native_effort'])
    mujoco.mj_step(model,d)
np.testing.assert_allclose(terminal[0],np.r_[d.qpos,d.qvel],rtol=1e-10,atol=1e-10)
timings=[]
for _ in range(20):
    t=time.perf_counter_ns();look.refine(q,v,target[0],reference);timings.append((time.perf_counter_ns()-t)*1e-6)
result=dict(held_target_state_error=float(np.abs(terminal[0]-np.r_[d.qpos,d.qvel]).max()),
    candidate_count=8,physics_steps_per_candidate=30,milliseconds_p50_p95_max=np.percentile(timings,[50,95,100]).tolist(),
    passed=True,full_motion_test_required=True)
(fw/'native_shoot_v1/physics_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
look.close()
