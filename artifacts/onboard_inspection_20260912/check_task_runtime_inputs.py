"""Compare training/runtime inputs on identical saved physical states.

This is an observation replay, not a physical rollout or acceptance result.
"""
from pathlib import Path
import json,sys
import numpy as np,mujoco
ROOT=Path(__file__).resolve().parents[2]
sys.path.append('/mnt/e/codex_sonic_runtime/bfm_seed_20260910/onnx_deps')
sys.path.append('/root/.venvs/g1_true23_mjlab/lib/python3.11/site-packages')
sys.path.insert(0,str(ROOT))
import torch
from gear_sonic.envs.mjlab.g1_true23_task_command_dynamics import TaskCommandEnv
from gear_sonic.envs.mjlab.g1_true23_causal_dynamics import quaternion_matrix
from gear_sonic.utils.g1_true23_task_commands import TaskCommandController
from gear_sonic.utils.g1_true23_bfmzero_stream import Packet

torch.set_num_threads(1)
FW=Path('/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/onboard_factory_firmware_v1')
BUNDLE=ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'
OLD=Path('/mnt/e/codex-artifacts/sonic23_teleop_six_hour_20260910')
PILOT=FW/'task_commands_ppo_v1'
env=TaskCommandEnv(BUNDLE,FW.parent/'causal_dynamics_v1/focused_walk002_task_closure_bank_v1',
    FW/'decoded_configs/policies/human_loco/fsm_human_loco_config.yaml',count=8,device='cpu',
    canonical_starts=True,canonical_worlds=6,command_delay_substeps=2,physics_backend='mjbatch')

class Capture:
    def __init__(self,session):self.session=session;self.x=None
    def run(self,names,inputs):
        self.x=inputs['features'][0].copy()
        return self.session.run(names,inputs)

results=[]
for clip in ('walk002','walk003','pico','walk008'):
    ci=next(i for i,r in enumerate(env.meta['clips']) if r['name']==clip)
    with np.load(OLD/f'mjbatch_intent_floor_inputs_v1/{clip}/reference.npz') as z:motion={k:z[k] for k in z.files}
    with np.load(BUNDLE/f'{clip}/original29.npz') as z:original={k:z[k] for k in z.files}
    with np.load(PILOT/f'eval_00100/{clip}/trace.npz') as z:trace={k:z[k] for k in ('qpos','qvel','target')}
    timeline=json.loads((BUNDLE/f'{clip}/timeline.json').read_text())
    teleop=TaskCommandController(PILOT/'actor_00100.onnx',FW,env.c,model=env.sim.model,
        tasks=env.meta['tasks'],standing_qpos=timeline['configured_standing_qpos'],now=-.22)
    capture=Capture(teleop.session);teleop.session=capture
    def receive(frame,now):
        fields={k:v[frame] for k,v in motion.items() if k!='fps'}
        assert teleop.receive(Packet(0,frame,frame*.02,fields),original['source_task_position_w'][frame],
            original['source_task_quaternion_wxyz'][frame],now)
    for frame in range(11):receive(frame,(frame-11)*.02)
    env.clips[:]=ci;env.frames[:]=11;env.age[:]=0;env.canonical_world[:]=True
    env.q[:]=env.states[ci,10,:30];env.v[:]=env.states[ci,10,30:]
    env.prior[:]=0
    for h in env.history:h[:]=0
    env.loco_previous[:]=0;env.command_velocity[:]=0;env.loco_phase[:]=0;env.loco_walking[:]=False
    env.loco_history[:]=env.native_prop()[:,None]
    peak=np.zeros(1582);target_error=0.
    for control in range(min(600,len(trace['qpos'])-1)):
        if control:
            q,v=trace['qpos'][control-1],trace['qvel'][control-1]
        else:q,v=env.states[ci,10,:30].numpy().astype(float),env.states[ci,10,30:].numpy().astype(float)
        # The batch only accepts state writes after an explicit reset. Without
        # this call forward() correctly restores the untouched native state.
        env.sim.reset(torch.arange(env.count))
        env.q[:]=torch.from_numpy(q);env.v[:]=torch.from_numpy(v);env.sim.forward()
        frame=control+11;receive(frame,control*.02)
        target=teleop.command(q,v,control*.02).targets
        x=env.observe()[0].numpy()
        peak=np.maximum(peak,np.abs(x-capture.x))
        target_error=max(target_error,float(np.max(np.abs(target-trace['target'][control]))))
        _,prop,velocity,phase,walking=env.control_fields()
        env.loco_history[:,:-1]=env.loco_history[:,1:].clone();env.loco_history[:,-1]=prop
        env.command_velocity[:]=velocity;env.loco_phase[:]=phase;env.loco_walking[:]=walking
        env.loco_previous[:]=torch.from_numpy(target[:12])-env.factory_default[:12]
        terms=(env.prior.clone(),env.v[:,3:6]*.25,env.q[:,7:]-env.default,env.v[:,6:],-quaternion_matrix(env.q[:,3:7])[:,2])
        for h,t in zip(env.history,terms):h[:,1:]=h[:,:-1].clone();h[:,0]=t
        env.prior[:]=(torch.from_numpy(target)-env.default)/env.action_scale
        env.age+=1;env.frames+=1;teleop.commit_applied(q,v,target)
    sections={name:float(peak[a:b].max()) for name,a,b in (
        ('native_history',0,210),('factory_command',210,218),('received_and_state',218,1541),
        ('closure_alpha',1541,1542),('body_errors',1542,1581),('history_valid',1581,1582))}
    row=dict(clip=clip,controls=control+1,max_abs_difference=sections,
        repeated_runtime_target_difference=target_error)
    assert max(sections.values())<1e-4,row
    assert target_error<1e-5,row
    results.append(row);print(json.dumps(row),flush=True)
out=FW/'task_commands_check_v1/runtime_inputs.json'
out.write_text(json.dumps(dict(cases=results,passed=True,observation_replay_only=True,physical_acceptance=False),indent=2))
