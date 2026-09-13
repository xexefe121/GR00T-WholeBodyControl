"""Check retained behavior and coherent latent exploration on real native states."""
from pathlib import Path
import sys,json
import numpy as np,mujoco
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from artifacts.teleop_resume_20260911.run_causal_native_clock import NEW
import torch,onnxruntime as ort
from gear_sonic.envs.mjlab.g1_true23_task_command_dynamics import TaskCommandEnv
from gear_sonic.utils.g1_true23_task_commands import TaskCommandActor

torch.set_num_threads(4)
fw=NEW/'onboard_factory_firmware_v1';cfg=fw/'decoded_configs/policies/human_loco/fsm_human_loco_config.yaml'
env=TaskCommandEnv(ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1',
    NEW/'causal_dynamics_v1/focused_walk002_task_closure_bank_v1',cfg,count=32,device='cpu',
    canonical_starts=True,canonical_worlds=6,command_delay_substeps=2,physics_backend='mjbatch')
actor=TaskCommandActor(fw/'human_loco_trainable_v1/factory_weights.npz',cfg,env.c['joint_limits'],
    fw/'native_mjbatch_lifecycle_ppo_v1/actor_00100.pt').eval()
x=env.observe().clone()
opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
previous=ort.InferenceSession(str(fw/'task_commands_check_v1/actor_zero.onnx'),sess_options=opts,providers=['CPUExecutionProvider'])
with torch.no_grad():
    means=actor.command_mean(x)
    assert means.shape==(32,17) and torch.count_nonzero(means)==0
    actual=actor.from_commands(x,means)
    torch.testing.assert_close(actual,actor(x),rtol=0,atol=0)
    old=previous.run(None,{'features':x.numpy()})[0]
    discrepancy=float(np.max(np.abs(old-actual.numpy())))
    assert discrepancy<5e-5,discrepancy
    effects=[]
    for index in range(12,17):
        sample=means.clone();sample[:,index]=.1
        changed=actor.from_commands(x,sample)
        effect=float((changed[:,:12]-actual[:,:12]).abs().max());effects.append(effect)
        assert effect>1e-5,(index,effect)
        torch.testing.assert_close(changed[:,12:],actual[:,12:],rtol=0,atol=0)
    sigma=torch.cat((.03/((actor.limits[:12,1]-actor.limits[:12,0])*.5),
        torch.full((2,),.3/np.pi),torch.tensor([.05/.6,.05/.5,.2/3.])))
    torch.manual_seed(20260912)
    sample=torch.distributions.Normal(means,sigma).sample()
    target=actor.target(actor.from_commands(x,sample))
    assert torch.isfinite(target).all()
    assert torch.all(target>=actor.limits[:,0]+.06-1e-6)
    assert torch.all(target<=actor.limits[:,1]-.06+1e-6)
    _,_,_,info=env.step(target)
result=dict(passed=True,previous_export_max_error=discrepancy,
    latent_action_dimensions=17,phase_velocity_effect_max_rad=effects,
    deterministic_upper_body_preserved=True,canonical_first_step_failures=int((info['failed']&env.canonical_world).sum()),
    full_motion_acceptance=False,hardware_commands=False)
out=fw/'command_space_check_v1';out.mkdir(exist_ok=False)
(out/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
