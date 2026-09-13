"""Check full retained controller prediction under received-only CV inputs."""
from pathlib import Path
import sys,json
import numpy as np,mujoco
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from artifacts.teleop_resume_20260911.run_causal_native_clock import NEW
from gear_sonic.scripts.evaluate_g1_true23_causal_dynamics import load_case,archive,assess
from gear_sonic.utils.g1_true23_locomotion_conditioned import LocomotionConditionedController
from gear_sonic.utils.g1_true23_feedback_planner import NativeFeedbackPlanner,predicted_received_goals,REFERENCE_FIELDS
from gear_sonic.utils.g1_true23_feedback_shoot import fptr
from gear_sonic.utils.g1_true23_native_shoot import ptr
from gear_sonic.utils.g1_true23_received_features import features_numpy
from gear_sonic.utils.g1_true23_bfmzero_stream import Packet
from scipy.spatial.transform import Rotation

fw=NEW/'onboard_factory_firmware_v1';folder=fw/'feedback_full_policy_planner_v1'
actor=fw/'received_pico_demo_v1/controller.onnx';checkpoint=fw/'native_mjbatch_lifecycle_ppo_v1/actor_00100.pt'
model,c,motion,original,timeline=load_case('pico');bank=NEW/'causal_dynamics_v1/bank'
tasks=json.loads((bank/'bank.json').read_text())['tasks'];state=archive(bank/'pico.npz')['states'][10]
controller=LocomotionConditionedController(actor,fw,c,model=model,tasks=tasks,
    standing_qpos=timeline['configured_standing_qpos'],now=-.22,target_filter_alpha=.9)
for seq in range(12):
    assert controller.receive(Packet(0,seq,seq*.02,{k:v[seq] for k,v in motion.items() if k!='fps'}),
        original['source_task_position_w'][seq],original['source_task_quaternion_wxyz'][seq],(seq-11)*.02)
q,v=state[:30].copy(),state[30:].copy();command=controller.command(q,v,0.);p=controller.policy
planner=NativeFeedbackPlanner(folder/'libtrue23shoot.so',model,c,p.kp,p.kd,tasks,200,
    firmware=fw,policy=p,learned_checkpoint=checkpoint,actor=actor)
feature_error=network_error=0.;samples=0
for clip in ('walk002','walk003','pico','walk008'):
    data=archive(bank/(clip+'.npz'))
    for frame in (11,370,450):
        reference={k:data[k][max(0,frame-38):frame+1] for k in REFERENCE_FIELDS}
        pos=np.ascontiguousarray(data['states'][frame,:30]);vel=np.ascontiguousarray(data['states'][frame,30:])
        predicted=predicted_received_goals(reference,20)
        f=features_numpy(pos,vel,reference,len(reference['joint'])-1,np.asarray(c['default_q']),np.zeros(23),np.zeros(300))
        native=np.empty(1000,np.float32)
        assert planner.lib.feedback_received_features(planner.context,ptr(pos),ptr(vel),ptr(predicted[0]),fptr(native))==0
        feature_error=max(feature_error,float(np.max(np.abs(native-f[:1000]))))
        h=np.ascontiguousarray(p.history,np.float32);cmd=np.ascontiguousarray(p.last_command,np.float32)
        output=np.empty(23,np.float32);f32=np.ascontiguousarray(f,np.float32)
        assert planner.lib.feedback_learned_raw(planner.context,fptr(h),fptr(cmd),fptr(f32),fptr(output))==0
        x=np.r_[h.ravel(),cmd,f32,0.].astype(np.float32)
        expected=controller.session.run(None,{'features':x[None]})[0][0]
        network_error=max(network_error,float(np.max(np.abs(output-expected))));samples+=1
assert feature_error<2e-5,feature_error
assert network_error<1e-4,network_error
print(json.dumps(dict(feature_error=feature_error,network_error=network_error)),flush=True)

# Same initial command, then full learned feedback at each20ms boundary. The
# hypothetical reference extends owned packets; this is not a source rollout.
reference=controller.last_reference
pack=predicted_received_goals(reference,20)
assert planner.lib.feedback_set_received_goals(planner.context,ptr(pack),len(pack))==0
r={k:a[-1] for k,a in reference.items()};quat=Rotation.from_matrix(r['root_rotation']).as_quat()[[3,0,1,2]]
goal=np.ascontiguousarray(np.r_[r['joint'],r['joint_velocity'],r['root'],r['root_velocity'],quat,r['root_omega'],
    r['feet'].ravel(),r['feet_velocity'].ravel(),r['tasks'].ravel(),r['task_velocity'].ravel(),r['task_rotation'].ravel(),r['task_omega'].ravel()])
h=np.ascontiguousarray(p.history,np.float32);cmd=np.ascontiguousarray(p.last_command,np.float32)
gait=np.ascontiguousarray(np.r_[p.phase,float(p.walking),cmd[5:]/.2]);candidate=np.ascontiguousarray(command.targets[None])
score=np.empty(1);terminal=np.empty((1,59))
assert planner.lib.feedback_evaluate(planner.context,ptr(q),ptr(v),ptr(candidate),1,200,ptr(goal),fptr(h),fptr(cmd),ptr(gait),ptr(score),ptr(terminal))==0
import ctypes as ct
from gear_sonic.utils.g1_true23_native_shoot import D
planner.lib.feedback_copy_trace.argtypes=[ct.c_void_p,D];planner.lib.feedback_copy_trace.restype=ct.c_int
native_trace=np.empty((30,82));trace_count=planner.lib.feedback_copy_trace(planner.context,ptr(native_trace))
saved_history=h.copy();saved_phase=p.phase;saved_walking=p.walking
extended={};offset=0
for key in REFERENCE_FIELDS:
    shape=np.asarray(reference[key]).shape[1:];width=int(np.prod(shape))
    future=pack[1:,0,offset:offset+width].reshape(20,*shape);offset+=width
    extended[key]=np.concatenate((reference[key],future))
owned=len(reference['joint']);stage=0
controller.receiver.reference=lambda now:{k:a[:owned+stage] for k,a in extended.items()}
data=mujoco.MjData(model);data.qpos[:]=q;data.qvel[:]=v;mujoco.mj_forward(model,data)
target=candidate[0].copy();controller.commit_applied(data.qpos,data.qvel,target)
band=np.minimum(.1,np.diff(c['joint_limits'],axis=1).ravel()*.2);failure=None;clock=0.
for step in range(200):
    if step and step%10==0:
        stage=step//10;target=controller.command(data.qpos,data.qvel,step*.002).targets
        controller.commit_applied(data.qpos,data.qvel,target)
    penetration=data.qpos[7:]-np.clip(data.qpos[7:],c['joint_limits'][:,0]+band,c['joint_limits'][:,1]-band)
    outward=np.where(penetration*data.qvel[6:]>0,data.qvel[6:],0)
    data.ctrl[:]=np.clip(p.kp*(target-data.qpos[7:])-p.kd*data.qvel[6:]-100*penetration-2*outward,-c['native_effort'],c['native_effort'])
    mujoco.mj_step(model,data);clock+=.002
    reasons,_=assess(data,c,clock)
    if reasons:failure=reasons;break
error=float(np.max(np.abs(terminal[0]-np.r_[data.qpos,data.qvel])))
controller.policy.history[:]=saved_history;controller.policy.phase=saved_phase;controller.policy.walking=saved_walking
controller.velocity[:]=gait[2:];controller.commit_applied(q,v,candidate[0])
oracle_errors=[]
for index in range(1,trace_count):
    stage=index;row=native_trace[index]
    actual=controller.command(row[:30],row[30:59],index*.02).targets
    oracle_errors.append(float(np.max(np.abs(actual-row[59:]))))
    controller.commit_applied(row[:30],row[30:59],row[59:])
print(json.dumps(dict(closed_loop_state_error=error,identical_state_target_errors=oracle_errors)),flush=True)
assert error<1e-3,error
result=dict(passed=True,feature_samples=samples,received_feature_max_error=feature_error,
    retained_onnx_max_error=network_error,full_feedback400ms_state_max_error=error,
    compared_physics_steps=step+1,predicted_failure=bool(score[0]>=1e9),ordinary_failure=failure,
    reference='owned history plus declared constant-velocity prediction',actual_future_source_frames=0,
    physical_acceptance=False,hardware_commands=False)
(folder/'check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
planner.close()
