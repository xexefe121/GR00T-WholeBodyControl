"""Exercise all23 learned joint commands on actual native simulator observations."""
from pathlib import Path
import copy,json,sys
import numpy as np,mujoco
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch,onnxruntime as ort
from artifacts.teleop_resume_20260911.run_causal_native_clock import NEW
from gear_sonic.envs.mjlab.g1_true23_task_command_dynamics import TaskCommandEnv
from gear_sonic.utils.g1_true23_task_commands import TaskCommandActor

torch.set_num_threads(4);torch.manual_seed(20260912)
fw=NEW/'onboard_factory_firmware_v1'
out=fw/'full_body_commands_check_v1';out.mkdir(exist_ok=False)
cfg=fw/'decoded_configs/policies/human_loco/fsm_human_loco_config.yaml'
env=TaskCommandEnv(ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1',
    NEW/'causal_dynamics_v1/focused_walk002_task_closure_bank_v1',cfg,count=32,device='cpu',
    canonical_starts=True,canonical_worlds=6,command_delay_substeps=2,physics_backend='mjbatch')
actor=TaskCommandActor(fw/'human_loco_trainable_v1/factory_weights.npz',cfg,env.c['joint_limits'],
    fw/'native_mjbatch_lifecycle_ppo_v1/actor_00100.pt',full_body_corrections=True).eval()
x=env.observe().clone()
opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
old=ort.InferenceSession(str(fw/'command_space_ppo_v1/actor_bootstrap.onnx'),sess_options=opts,providers=['CPUExecutionProvider'])
with torch.no_grad():
    mean=actor.command_mean(x);assert mean.shape==(32,28) and torch.count_nonzero(mean)==0
    nominal=actor(x)
    baseline_error=float(np.max(np.abs(old.run(None,{'features':x.numpy()})[0]-nominal.numpy())))
    assert baseline_error<5e-5,baseline_error
    effects=[]
    for j in range(23):
        command=mean.clone();command[:,j]=.01
        changed=actor.from_commands(x,command)-nominal
        effect=float(changed[:,j].abs().max());effects.append(effect)
        assert effect>1e-5,(j,effect)
        other=torch.cat((changed[:,:j],changed[:,j+1:]),1)
        assert torch.count_nonzero(other)==0,j
    phase_velocity=[]
    for j in range(23,28):
        command=mean.clone();command[:,j]=.1
        changed=actor.from_commands(x,command)-nominal
        effect=float(changed[:,:12].abs().max());phase_velocity.append(effect)
        assert effect>1e-5,(j,effect)
        assert torch.count_nonzero(changed[:,12:])==0

# Check that upper-body corrections participate in an actual optimizer update.
# The discarded copy keeps baseline export and physical preflight unchanged.
probe=copy.deepcopy(actor).train();optimizer=torch.optim.Adam(probe.goal_head.parameters(),lr=3e-5)
desired=nominal.detach().clone();desired[:,12:]+=.01
loss=(probe(x)-desired).square().mean();optimizer.zero_grad();loss.backward()
upper_gradient=float(probe.goal_head[-1].weight.grad[12:23].abs().max())
assert upper_gradient>0;optimizer.step()
with torch.no_grad():
    upper_update=float((probe(x)[:,12:]-nominal[:,12:]).abs().max())
assert upper_update>0
path=out/'actor_zero.onnx'
torch.onnx.export(actor,torch.zeros(1,1582),str(path),input_names=['features'],output_names=['normalized_target'],
    dynamic_axes={'features':{0:'batch'},'normalized_target':{0:'batch'}},opset_version=17,dynamo=False)
session=ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])
parity=float(np.max(np.abs(session.run(None,{'features':x.numpy()})[0]-nominal.detach().numpy())))
assert parity<5e-5,parity
with torch.no_grad():
    sigma=torch.cat((.03/((actor.limits[:,1]-actor.limits[:,0])*.5),
        torch.full((2,),.3/np.pi),torch.tensor([.05/.6,.05/.5,.2/3.])))
    command=torch.distributions.Normal(mean,sigma).sample()
    target=actor.target(actor.from_commands(x,command))
    _,_,_,info=env.step(target)
    failures=int((info['failed']&env.canonical_world).sum())
    assert failures==0,failures
result=dict(passed=True,learned_joint_targets=23,latent_commands=28,features=1582,
    original_baseline_max_error=baseline_error,onnx_max_error=parity,
    joint_command_effects_rad=effects,phase_velocity_effects_rad=phase_velocity,
    upper_output_weight_gradient_max=upper_gradient,upper_target_update_max=upper_update,
    sampled_canonical_first_step_failures=failures,
    original_acceptance_limits=True,physical_full_motion_passed=False,hardware_commands=False)
(out/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
