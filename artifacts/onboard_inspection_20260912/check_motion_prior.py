"""Check physical transition semantics and measure the local simulator pilot."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
if mujoco.__version__=='3.2.3' and sys.platform!='win32':
    for dependency in ('/mnt/e/codex_sonic_runtime/bfm_seed_20260910/onnx_deps',
                       '/root/.venvs/g1_true23_mjlab/lib/python3.11/site-packages'):
        sys.path.append(dependency)
import torch
from scipy.spatial.transform import Rotation
from gear_sonic.envs.mjlab.g1_true23_task_command_dynamics import TaskCommandEnv
from gear_sonic.utils.g1_true23_task_commands import TaskCommandActor
from gear_sonic.utils.g1_true23_motion_prior import MotionPrior,physical_features


def run(args):
    torch.manual_seed(20260913);torch.set_num_threads(4 if args.device=='cpu' else 1)
    args.output.mkdir(exist_ok=False)
    base=Path('E:/codex-artifacts' if sys.platform=='win32' else '/mnt/e/codex-artifacts')/'sonic23_teleop_resume_20260911'
    fw=base/'onboard_factory_firmware_v1'
    bank=base/'causal_dynamics_v1/focused_walk002_task_closure_bank_v1'
    cfg=fw/'decoded_configs/policies/human_loco/fsm_human_loco_config.yaml'
    started=time.monotonic()
    env=TaskCommandEnv(ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1',bank,cfg,
        count=args.count,device=args.device,canonical_starts=True,canonical_worlds=4,
        command_delay_substeps=2,physics_backend=args.backend)
    env.terminate_tracking_errors=False;env.capture_motion_prior=True
    actor=TaskCommandActor(fw/'human_loco_trainable_v1/factory_weights.npz',cfg,env.c['joint_limits'],
        fw/'native_mjbatch_lifecycle_ppo_v1/actor_00100.pt',full_body_corrections=True).to(args.device).eval()
    prior=MotionPrior(fw/'motion_prior_data_v1/transitions.npz',args.device)
    # Rotate and translate a full physical scene; local dynamics features must
    # stay unchanged. Angular qvel remains body-local under world-yaw rotation.
    q=env.q.detach().cpu().numpy().astype(np.float64);v=env.v.detach().cpu().numpy().astype(np.float64)
    feet=env.sim.data.xpos[:,env.feet].cpu().numpy().astype(np.float64)
    mats=env.sim.data.xmat[:,env.task_ids].reshape(args.count,3,3,3)
    tasks=(env.sim.data.xpos[:,env.task_ids]+torch.einsum('ntij,tj->nti',mats,env.task_offsets)).cpu().numpy().astype(np.float64)
    old=physical_features(*[torch.tensor(a) for a in (q,v,feet,tasks)])
    rot=Rotation.from_euler('z',1.23);matrix=rot.as_matrix();shift=np.array([3.,-2.,0.])
    changed=q.copy();changed[:,:3]=q[:,:3]@matrix.T+shift
    changed[:,3:7]=(rot*Rotation.from_quat(q[:,[4,5,6,3]])).as_quat()[:,[3,0,1,2]]
    velocity=v.copy();velocity[:,:3]=v[:,:3]@matrix.T
    transformed=physical_features(*[torch.tensor(a) for a in
        (changed,velocity,feet@matrix.T+shift,tasks@matrix.T+shift)])
    invariant_error=float((old-transformed).abs().max());assert invariant_error<2e-6,invariant_error
    # A falling terminal successor must be retained, not replaced by reset.
    saved_q,saved_v=env.q[0].clone(),env.v[0].clone()
    env.sim.reset(torch.tensor([0],device=args.device))
    env.q[0]=saved_q;env.v[0]=saved_v
    env.q[0,2]=.1;env.sim.forward()
    with torch.no_grad():
        _,_,done,info=env.step(actor.target(actor(env.observe())))
    assert bool(done[0]) and bool(info['failed'][0])
    terminal_height=float(info['motion_prior_next'][0,0]);reset_height=float(env.q[0,2])
    assert terminal_height<.25 and reset_height>.25,(terminal_height,reset_height)
    pair=torch.cat((info['motion_prior_next'],info['motion_prior_next']),1)
    assert float(prior.reward(pair,info['failed'])[0])==0
    env.reset(torch.arange(env.count,device=args.device))
    if args.device!='cpu':torch.cuda.synchronize()
    setup=time.monotonic()-started;tick=time.monotonic();pairs=[];failures=0
    with torch.no_grad():
        for _ in range(64):
            before=env.motion_prior_observation()
            _,_,_,info=env.step(actor.target(actor(env.observe())))
            pairs.append(torch.cat((before,info['motion_prior_next']),1))
            failures+=int(info['failed'].sum())
    if args.device!='cpu':torch.cuda.synchronize()
    elapsed=time.monotonic()-tick
    real=prior.sample_expert(512);fake=real+torch.randn_like(real)*prior.scale*2
    for _ in range(8):prior.update(fake,steps=8)
    with torch.no_grad():
        valid=torch.zeros(len(real),device=args.device,dtype=torch.bool)
        real_reward=float(prior.reward(real,valid).mean());fake_reward=float(prior.reward(fake,valid).mean())
    assert real_reward>fake_reward+.1,(real_reward,fake_reward)
    stats=prior.update(torch.cat(pairs),steps=8)
    result=dict(passed=True,backend=args.backend,mujoco=mujoco.__version__,device=args.device,count=args.count,
        setup_seconds=setup,physics_rollout_seconds=elapsed,controlled_states_per_second=64*args.count/elapsed,
        scene_yaw_translation_invariance_max_error=invariant_error,terminal_height=terminal_height,
        reset_height=reset_height,failed_transition_style_reward=0,expert_reward=real_reward,
        perturbed_reward=fake_reward,discriminator_actual_rollout=stats,benchmark_failed_world_steps=failures,
        original_actor_width=1582,actor_outputs=23,full_motion_tested=False)
    if args.device!='cpu':
        result['cuda_memory_allocated_bytes']=torch.cuda.max_memory_allocated()
        result['cuda_free_total_bytes']=torch.cuda.mem_get_info()
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--backend',default='mjbatch')
    p.add_argument('--count',type=int,default=16);run(p.parse_args())
