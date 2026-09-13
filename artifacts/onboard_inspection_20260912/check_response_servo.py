"""Compare predicted native state/point velocity with ordinary MuJoCo."""
from pathlib import Path
import sys,json,time
import numpy as np,mujoco
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from artifacts.teleop_resume_20260911.run_causal_native_clock import NEW
from gear_sonic.scripts.evaluate_g1_true23_causal_dynamics import load_case,archive
from gear_sonic.utils.g1_true23_response_servo import NativeResponseServo
from gear_sonic.utils.g1_true23_received_features import prepare_reference
from gear_sonic.utils.g1_true23_factory_locomotion import IDS
import yaml

fw=NEW/'onboard_factory_firmware_v1';model,c,motion,original,timeline=load_case('walk002')
meta=json.loads((NEW/'causal_dynamics_v1/bank/bank.json').read_text())
cfg=yaml.safe_load((fw/'decoded_configs/policies/human_loco/fsm_human_loco_config.yaml').read_text(encoding='utf-8'))
kp=np.asarray(cfg['joint_kp'])[IDS];kd=np.asarray(cfg['joint_kd'])[IDS]
initial=archive(NEW/'causal_dynamics_v1/bank/walk002.npz')['states'][10]
servo=NativeResponseServo(fw/'response_servo_v1/libtrue23response.so',model,c,kp,kd,meta['tasks'],10)
q,v=initial[:30],initial[30:];target=initial[7:30].copy()
out,bad=servo.predict(q,v,target[None]);assert not bad.any()
d=mujoco.MjData(model);d.qpos[:]=q;d.qvel[:]=v;mujoco.mj_forward(model,d)
for _ in range(10):
    band=np.minimum(.1,np.diff(c['joint_limits'],axis=1).ravel()*.2)
    penetration=d.qpos[7:]-np.clip(d.qpos[7:],c['joint_limits'][:,0]+band,c['joint_limits'][:,1]-band)
    outward=np.where(penetration*d.qvel[6:]>0,d.qvel[6:],0)
    bounded=np.clip(target,c['joint_limits'][:,0]+.06,c['joint_limits'][:,1]-.06)
    d.ctrl[:]=np.clip(kp*(bounded-d.qpos[7:])-kd*d.qvel[6:]-100*penetration-2*outward,-c['native_effort'],c['native_effort'])
    mujoco.mj_step(model,d)
mujoco.mj_forward(model,d)
points=[];velocities=[]
for body,offset in zip(servo.ids,[[0,0,0],[0,0,0]]+[t['target_point'] for t in meta['tasks']]):
    point=d.xpos[body]+d.xmat[body].reshape(3,3)@offset;jac=np.zeros((3,29))
    mujoco.mj_jac(model,d,jac,None,point,int(body));points.append(point);velocities.append(jac@d.qvel)
expected=np.r_[d.qpos,d.qvel,np.array(points).ravel(),np.array(velocities).ravel()]
error=float(np.max(np.abs(expected-out[0])));assert error<1e-9,error
reference=prepare_reference({k:a[:12] for k,a in motion.items() if k!='fps'},
    {k:a[:12] for k,a in original.items() if k.startswith('source_task_')},c)
samples=[]
for _ in range(20):
    start=time.perf_counter_ns();result=servo.refine(q,v,target,reference);samples.append((time.perf_counter_ns()-start)*1e-6)
    assert result.shape==(23,) and np.isfinite(result).all()
    assert (result>=servo.lower).all() and (result<=servo.upper).all()
report=dict(passed=True,predicted_state_point_velocity_max_error=error,
    response_ms_p50_p95_max=np.percentile(samples[2:],[50,95,100]).tolist(),last=servo.status,
    actual_physics_modified=False,complete_motion_test_required=True)
(fw/'response_servo_v1/physics_check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
servo.close()
