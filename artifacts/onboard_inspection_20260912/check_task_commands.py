"""Check real FK features, preserved initialization, causal inputs and export."""
from pathlib import Path
import sys,json
import numpy as np,mujoco
ROOT=Path(__file__).resolve().parents[2]
sys.path.append('/mnt/e/codex_sonic_runtime/bfm_seed_20260910/onnx_deps')
sys.path.append('/root/.venvs/g1_true23_mjlab/lib/python3.11/site-packages')
sys.path.insert(0,str(ROOT))
import torch,onnxruntime as ort
from gear_sonic.envs.mjlab.g1_true23_task_command_dynamics import TaskCommandEnv
from gear_sonic.utils.g1_true23_task_commands import TaskCommandActor,task_errors_numpy
torch.set_num_threads(4)
FW=Path('/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/onboard_factory_firmware_v1')
cfg=FW/'decoded_configs/policies/human_loco/fsm_human_loco_config.yaml'
env=TaskCommandEnv(ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1',
    FW.parent/'causal_dynamics_v1/focused_walk002_task_closure_bank_v1',cfg,count=32,device='cpu',
    canonical_starts=True,canonical_worlds=6,command_delay_substeps=2,physics_backend='mjbatch')
env.terminate_tracking_errors=False
actor=TaskCommandActor(FW/'human_loco_trainable_v1/factory_weights.npz',cfg,env.c['joint_limits'],
    FW/'native_mjbatch_lifecycle_ppo_v1/actor_00100.pt').eval()
x=env.observe();assert x.shape==(32,1582)
errors=[]
for i in range(32):
    clip=int(env.clips[i]);frame=int(env.frames[i])
    refs={k:v[clip,:frame+1].numpy() for k,v in env.references.items()}
    value=task_errors_numpy(env.sim.model,mujoco.MjData(env.sim.model),env.q[i].numpy(),env.v[i].numpy(),refs,env.meta['tasks'])
    errors.append(float(np.max(np.abs(value-x[i,1542:1581].numpy()))))
assert max(errors)<2e-5,errors
with torch.no_grad():
    original=actor.base.target(actor.base(x[:,:1542]));previous=actor.default[:12]+x[:,:210].reshape(-1,5,42)[:,-1,30:42]
    original[:,:12]=previous+(1-.1*x[:,-1:])*(original[:,:12]-previous)
    preserved=float((actor.target(actor(x))-original).abs().max())
    assert preserved<2e-5,preserved
    before=actor(x).clone()
    actor.goal_head[-1].bias[12]=.1
    effect=float((actor(x)-before).abs().max());assert effect>1e-4
    actor.goal_head[-1].bias.zero_()
    baseline=env.observe().clone()
    saved={k:v.clone() for k,v in env.references.items()}
    # Worlds can share a recording but have different frame ages. Check one
    # representative world per recording and mutate only its unseen suffix.
    for clip in env.clips.unique():
        i=int(torch.nonzero(env.clips==clip)[0]);frame=int(env.frames[i])
        for v in env.references.values():v[clip,frame+1:]=123.456
        torch.testing.assert_close(env.observe()[i],baseline[i],rtol=0,atol=0)
        for key,v in env.references.items():v.copy_(saved[key])
out=FW/'task_commands_check_v1';out.mkdir(exist_ok=True)
torch.onnx.export(actor,x[:1],str(out/'actor_zero.onnx'),input_names=['features'],output_names=['normalized_target'],
    dynamic_axes={'features':{0:'batch'},'normalized_target':{0:'batch'}},opset_version=17,dynamo=False)
opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
session=ort.InferenceSession(str(out/'actor_zero.onnx'),sess_options=opts,providers=['CPUExecutionProvider'])
with torch.no_grad():
    actual=actor(x).numpy();expected=session.run(None,{'features':x.numpy()})[0]
    exported=float(np.max(np.abs(actual-expected)));assert exported<5e-5,exported
    _,_,_,info=env.step(actor.target(actor(x)))
result=dict(passed=True,features=1582,native_fk_feature_error=max(errors),preserved_target_error=preserved,
    learned_phase_changes_targets=effect,onnx_error=exported,future_suffix_changes_inputs=False,
    canonical_worlds_physical_failures=int((info['failed']&env.canonical_world).sum()),hardware_commands=False)
(out/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
