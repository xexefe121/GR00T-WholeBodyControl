"""Check factory inference, first-command integration and feedback prediction."""
from pathlib import Path
import sys,json,time,argparse
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from artifacts.teleop_resume_20260911.run_causal_native_clock import NEW
from gear_sonic.scripts.evaluate_g1_true23_causal_dynamics import load_case,archive
from gear_sonic.utils.g1_true23_locomotion_conditioned import LocomotionConditionedController
from gear_sonic.utils.g1_true23_factory_locomotion import FactoryHumanLoco12
from gear_sonic.utils.g1_true23_bfmzero_stream import Packet
from gear_sonic.utils.g1_true23_native_shoot import NativePoseLookahead
from gear_sonic.utils.g1_true23_feedback_shoot import NativeFeedbackLookahead,fptr

fw=NEW/'onboard_factory_firmware_v1'
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=fw/'native_feedback_shoot_v1');args=ap.parse_args()
out=args.output;lib=out/'libtrue23shoot.so'
model,c,motion,original,timeline=load_case('walk002');bank=NEW/'causal_dynamics_v1/bank'
tasks=json.loads((bank/'bank.json').read_text())['tasks'];state=archive(bank/'walk002.npz')['states'][10]
controller=LocomotionConditionedController(fw/'locomotion_conditioned_ppo_v2/actor_00100.onnx',fw,c,model=model,tasks=tasks,
    standing_qpos=timeline['configured_standing_qpos'],now=-.22)
for seq in range(12):
    assert controller.receive(Packet(0,seq,seq*.02,{k:v[seq] for k,v in motion.items() if k!='fps'}),
        original['source_task_position_w'][seq],original['source_task_quaternion_wxyz'][seq],(seq-11)*.02)
q=state[:30];v=state[30:];command=controller.command(q,v,0.);p=controller.policy
feedback=NativeFeedbackLookahead(lib,model,c,p.kp,p.kd,tasks,10,firmware=fw,policy=p)
fixed=NativePoseLookahead(lib,model,c,p.kp,p.kd,tasks,steps=10)
rng=np.random.default_rng(20260913);errors=[]
for _ in range(128):
    h=rng.normal(size=(5,42)).astype(np.float32);cmd=rng.normal(size=8).astype(np.float32);raw=np.empty(12,np.float32)
    feedback.lib.feedback_raw(feedback.context,fptr(h),fptr(cmd),fptr(raw))
    wanted=p.onnx.run(['act'],{'p_obs':h[None],'cmd':cmd[None]})[0][0]
    errors.append(float(np.max(np.abs(raw-wanted))))
assert max(errors)<1e-5,max(errors)
feedback.refine(q,v,command.targets,controller.last_reference);fixed.refine(q,v,command.targets,controller.last_reference)
np.testing.assert_array_equal(feedback.candidates,fixed.candidates)
np.testing.assert_array_equal(feedback.terminals,fixed.terminals)
np.testing.assert_array_equal(feedback.scores,fixed.scores)

# Independent ordinary Python/MuJoCo rollout of candidate0: preserve the same
# recovered balance history, update it every20ms, and hold candidate leg offset.
feedback.steps=200;feedback.refine(q,v,command.targets,controller.last_reference)
native=feedback.terminals[0].copy();r={k:x[-1] for k,x in controller.last_reference.items()}
pp=FactoryHumanLoco12(fw,c['joint_limits'],fw/'human_loco_trainable_v1/factory_loco12.onnx')
pp.history[:]=p.history;pp.initialized=True;pp.phase=p.phase;pp.walking=p.walking
vel=p.last_command[5:].astype(np.float64)/.2
raw=p.onnx.run(['act'],{'p_obs':p.history[None],'cmd':p.last_command[None]})[0][0]
residual=feedback.candidates[0,:12]-(p.default[:12]+p.scale*np.clip(raw,-10,10))
d=mujoco.MjData(model);d.qpos[:]=q;d.qvel[:]=v;mujoco.mj_forward(model,d)
target=feedback.candidates[0].copy();band=np.minimum(.1,np.diff(c['joint_limits'],axis=1).ravel()*.2)
for step in range(200):
    if step and step%10==0:
        t=step*.002;rot=np.empty(9);mujoco.mju_quat2Mat(rot,d.qpos[3:7]);rot=rot.reshape(3,3)
        yaw=np.arctan2(rot[1,0],rot[0,0]);cy,sy=np.cos(yaw),np.sin(yaw)
        xy=r['root_velocity'][:2]+2.5*(r['root'][:2]+t*r['root_velocity'][:2]-d.qpos[:2])
        gy=np.arctan2(r['root_rotation'][1,0],r['root_rotation'][0,0])+t*r['root_omega'][2]
        dy=np.arctan2(np.sin(gy-yaw),np.cos(gy-yaw))
        wanted=np.clip(np.r_[xy@np.array([[cy,-sy],[sy,cy]]),r['root_omega'][2]+3*dy],[-1.2,-1,-6],[1.2,1,6])
        vel+=np.clip(wanted-vel,[-.24,-.24,-.8],[.24,.24,.8])
        pp.previous[:]=target[:12]-pp.default[:12]
        hist,cmd=pp.observation_command(d.qpos,d.qvel,-rot[2],vel)
        rr=pp.onnx.run(['act'],{'p_obs':hist[None],'cmd':cmd[None]})[0][0]
        target=np.r_[pp.default[:12]+pp.scale*np.clip(rr,-10,10)+residual,
            feedback.candidates[0,12:]+t*r['joint_velocity'][12:]]
        target=np.clip(target,c['joint_limits'][:,0]+.06,c['joint_limits'][:,1]-.06)
    penetration=d.qpos[7:]-np.clip(d.qpos[7:],c['joint_limits'][:,0]+band,c['joint_limits'][:,1]-band)
    outward=np.where(penetration*d.qvel[6:]>0,d.qvel[6:],0)
    d.ctrl[:]=np.clip(p.kp*(target-d.qpos[7:])-p.kd*d.qvel[6:]-100*penetration-2*outward,-c['native_effort'],c['native_effort'])
    mujoco.mj_step(model,d)
prediction_error=float(np.max(np.abs(native-np.r_[d.qpos,d.qvel])))
assert prediction_error<2e-4,prediction_error
timing=[]
for _ in range(12):
    t=time.perf_counter();feedback.refine(q,v,command.targets,controller.last_reference);timing.append((time.perf_counter()-t)*1000)
result=dict(passed=True,factory_random_samples=128,factory_onnx_max_error=max(errors),
    first20ms_held_target_states_exact=True,feedback400ms_state_max_error=prediction_error,
    prediction_ms_p50_p95_max=np.percentile(timing[2:],[50,95,100]).tolist(),
    timing_qualification=False,training_may_be_running=True,physical_limits_unchanged=True,future_received_frames=0)
(out/'check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
feedback.close();fixed.close()
