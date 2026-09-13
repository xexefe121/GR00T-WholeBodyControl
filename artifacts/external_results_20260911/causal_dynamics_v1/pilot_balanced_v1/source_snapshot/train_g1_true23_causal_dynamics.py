"""Bounded full-range, received-reference native23 PPO pilot. Simulation only."""
import argparse
import copy
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from torch import nn
from gear_sonic.envs.mjlab.g1_true23_causal_dynamics import Actor, CausalNative23Env, features_torch
from gear_sonic.utils.g1_true23_received_features import features_numpy

ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911')


def write(path,value): path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def check_features(env):
    actual=env.observe().cpu().numpy()
    for i in range(min(8,env.count)):
        clip=int(env.clips[i]);frame=int(env.frames[i])
        refs={k:v[clip,:env.lengths[clip]].cpu().numpy() for k,v in env.references.items()}
        expected=features_numpy(env.q[i].cpu().numpy(),env.v[i].cpu().numpy(),refs,frame,
            env.default.cpu().numpy(),env.prior[i].cpu().numpy(),env.history_vector()[i].cpu().numpy())
        np.testing.assert_allclose(actual[i],expected,rtol=2e-5,atol=2e-5)
        # Mutating every unseen future reference must leave inputs unchanged.
        changed={k:v.copy() for k,v in refs.items()}
        for v in changed.values(): v[frame+1:]=123.456
        altered=features_numpy(env.q[i].cpu().numpy(),env.v[i].cpu().numpy(),changed,frame,
            env.default.cpu().numpy(),env.prior[i].cpu().numpy(),env.history_vector()[i].cpu().numpy())
        np.testing.assert_array_equal(expected,altered)
    return dict(states=min(8,env.count),cpu_gpu_features_match=True,future_mutation_has_no_effect=True)


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(20260912)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.monotonic()
    saved=torch.load(args.initial,map_location='cpu',weights_only=False)
    actor=Actor(saved).cuda()
    critic=nn.Sequential(nn.Linear(1323,256),nn.ELU(),nn.Linear(256,256),nn.ELU(),nn.Linear(256,1)).cuda()
    if args.resume is not None:
        resumed=torch.load(args.resume,map_location='cuda:0',weights_only=False)
        actor.load_state_dict(resumed['actor']);critic.load_state_dict(resumed['critic'])
    optimizer=torch.optim.Adam([{'params':actor.parameters(),'lr':3e-6},{'params':critic.parameters(),'lr':3e-4}])
    env_class=CausalNative23Env
    if args.balanced:
        from gear_sonic.envs.mjlab.g1_true23_balanced_causal_dynamics import BalancedCausalNative23Env
        env_class=BalancedCausalNative23Env
    env=env_class(ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1',args.bank,args.num_envs)
    write(args.output/'input_checks.json',check_features(env))
    # Recondition the changed causal observations while preserving the initial
    # actor function. This avoids huge gradients on newly shifted features.
    if args.resume is None:
        with torch.no_grad():
            new_mean=env.expert_features.mean(0)
            new_scale=env.expert_features.std(0).clamp_min(.1)
            x=env.expert_features[:128];before=actor(x).clone()
            layer=actor.net[0];old_weight=layer.weight.clone()
            layer.bias.add_(old_weight@((new_mean-actor.mean)/actor.scale))
            layer.weight.mul_(new_scale/actor.scale)
            actor.mean.copy_(new_mean);actor.scale.copy_(new_scale)
            error=float(((actor(x)-before)*actor.span).abs().max())
            if error>1e-4:raise ValueError('normalization reparameterization changed initial targets')
            write(args.output/'normalization_change.json',dict(max_target_error_rad=error))
    request=dict(kind='full_range_causal_native23_dynamics_ppo',initial_checkpoint=str(args.initial),
        resume_actor_and_critic=str(args.resume) if args.resume else None,optimizer_reinitialized=bool(args.resume),
        retained_neutral_balance=args.balanced,motion_takeover_seconds=.5 if args.balanced else 0,
        actor_width=512,features=1323,future_reference_frames=0,reference_derivatives='backward',
        native_pd=True,physics_hz=500,control_hz=50,simulator='MJLab MuJoCo-Warp 3.5.0',
        final_referee='MuJoCo3.2.3',num_envs=args.num_envs,rollout_steps=args.steps,
        updates=args.updates,bootstrap_updates=args.bootstrap_updates,wall_hours=args.wall_hours,
        milestones=[args.updates//2,args.updates],heldout='walk008',root_state='simulation_privileged',
        reset_schedule='50pct canonical lifecycle;50pct actual expert states;root XY velocity +/-0.03mps',
        hardware_authorized=False,simulation_qualified=False)
    if args.balanced:
        request['reset_schedule']='equal canonical starts, actual expert states, actual states near six interruption points; XY velocity +/-0.03mps'
    write(args.output/'request.json',request)
    print(json.dumps(dict(ready=True,**request)),flush=True)

    def save(label,update):
        path=args.output/f'actor_{label}.pt'
        checkpoint=dict(actor=actor.state_dict(),critic=critic.state_dict(),optimizer=optimizer.state_dict(),
            source= str(args.initial),request=request,update=update,elapsed_s=time.monotonic()-started)
        torch.save(checkpoint,path)
        cpu=copy.deepcopy(actor).cpu().eval()
        torch.onnx.export(cpu,torch.zeros(1,1323),str(path.with_suffix('.onnx')),input_names=['features'],
            output_names=['normalized_target'],dynamic_axes={'features':{0:'batch'},'normalized_target':{0:'batch'}},
            opset_version=17,dynamo=False)
        np.savez(path.with_suffix('.normalization.npz'),span=cpu.span.numpy(),default=cpu.default.numpy(),limits=cpu.limits.numpy())
        print(json.dumps(dict(checkpoint=str(path),update=update)),flush=True)

    save('initial',0)
    # Short adaptation to changed observation semantics, then physics drives
    # optimization. Clips are sampled uniformly, not dominated by long Pico.
    perclip=[(env.expert_resets[:,0]==i).nonzero().flatten() for i in range(3)]
    boot=torch.optim.Adam(actor.parameters(),lr=3e-5)
    def expert_batch(n=256):
        ids=torch.cat([idx[torch.randint(len(idx),(n//3+1,),device='cuda')] for idx in perclip])[:n]
        return env.expert_features[ids],(env.expert_targets[ids]-actor.default)/actor.span
    for i in range(args.bootstrap_updates):
        x,y=expert_batch()
        pred=actor(x);loss=((pred-y)*actor.span).square().mean()
        boot.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(actor.parameters(),1);boot.step()
        if i%200==0: print(json.dumps(dict(bootstrap=i,rmse_rad=float(loss.detach().sqrt()),elapsed_s=time.monotonic()-started)),flush=True)
    save('bootstrap',0)
    if args.wait_initial_evaluation:
        print('Waiting for initial full-motion evaluation before PPO.',flush=True)
        deadline=time.monotonic()+600
        while not (args.output/'CONTINUE').exists():
            if time.monotonic()>deadline:raise TimeoutError('initial evaluation release absent')
            time.sleep(1)
    obs=env.observe()
    std=.03/actor.span
    gamma,lam=.995,.95
    count=args.steps*args.num_envs
    buffers={k:torch.empty(shape,device='cuda') for k,shape in dict(
        obs=(args.steps,args.num_envs,1323),action=(args.steps,args.num_envs,23),
        mean=(args.steps,args.num_envs,23),logprob=(args.steps,args.num_envs),value=(args.steps,args.num_envs),
        reward=(args.steps,args.num_envs),done=(args.steps,args.num_envs),policy_weight=(args.steps,args.num_envs)).items()}
    rollout_started=time.monotonic();completed=0
    with (args.output/'metrics.jsonl').open('x') as log:
        for update in range(1,args.updates+1):
            tick=time.monotonic();failures=successes=0;age_sum=[];root=[];legs=[];active=[]
            with torch.no_grad():
                for t in range(args.steps):
                    mean=actor(obs)
                    distribution=torch.distributions.Normal(mean,std)
                    action=distribution.sample()
                    buffers['obs'][t]=obs;buffers['action'][t]=action;buffers['mean'][t]=mean
                    buffers['logprob'][t]=distribution.log_prob(action).sum(-1)
                    buffers['value'][t]=critic((obs-actor.mean)/actor.scale).squeeze(-1)
                    obs,reward,done,info=env.step(actor.target(action))
                    buffers['reward'][t]=reward;buffers['done'][t]=done
                    weight=(info['motion_fraction']>0).float() if 'motion_fraction' in info else torch.ones_like(reward)
                    buffers['policy_weight'][t]=weight;active.append(weight.mean())
                    failures+=int(info['failed'].sum());successes+=int(info['complete'].sum())
                    if done.any():age_sum.extend(info['age'][done].cpu().tolist())
                    root.append(info['root_error'].mean());legs.append(info['leg_error'].mean())
                next_value=critic((obs-actor.mean)/actor.scale).squeeze(-1)
                advantage=torch.zeros_like(buffers['reward']);carry=torch.zeros_like(next_value)
                for t in range(args.steps-1,-1,-1):
                    alive=1-buffers['done'][t]
                    delta=buffers['reward'][t]+gamma*next_value*alive-buffers['value'][t]
                    carry=delta+gamma*lam*alive*carry;advantage[t]=carry;next_value=buffers['value'][t]
                returns=(advantage+buffers['value']).flatten()
                adv=advantage.flatten();adv=(adv-adv.mean())/(adv.std()+1e-8)
            flat={k:v.flatten(0,1) for k,v in buffers.items()}
            losses=[];maxkl=0.;rejected=0
            for epoch in range(4):
                for ix in torch.randperm(count,device='cuda').chunk(4):
                    x=flat['obs'][ix];mean=actor(x)
                    dist=torch.distributions.Normal(mean,std)
                    lp=dist.log_prob(flat['action'][ix]).sum(-1)
                    delta=lp-flat['logprob'][ix];ratio=delta.exp()
                    weight=flat['policy_weight'][ix]
                    active_batch=bool(weight.any())
                    policy=-(torch.minimum(ratio*adv[ix],ratio.clamp(.8,1.2)*adv[ix])*weight).sum()/weight.sum().clamp_min(1)
                    value=critic((x-actor.mean)/actor.scale).squeeze(-1)
                    value_loss=(value-returns[ix]).square().mean()
                    bx,by=expert_batch(128)
                    imitation=((actor(bx)-by)*actor.span).square().mean()
                    loss=policy+.5*value_loss+.2*imitation
                    if not torch.isfinite(loss):raise RuntimeError('nonfinite PPO loss')
                    old_state=copy.deepcopy(actor.state_dict())
                    old_optimizer={p:copy.deepcopy(optimizer.state.get(p,{})) for p in actor.parameters()}
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    if not active_batch:
                        for parameter in actor.parameters():parameter.grad=None
                    nn.utils.clip_grad_norm_(list(actor.parameters())+list(critic.parameters()),.5);optimizer.step()
                    with torch.no_grad():
                        per_state_kl=.5*((actor(x)-flat['mean'][ix])/std).square().sum(-1)
                        kl=float((per_state_kl*weight).sum()/weight.sum().clamp_min(1))
                    if not np.isfinite(kl) or kl>.03:
                        actor.load_state_dict(old_state)
                        for parameter,state in old_optimizer.items():optimizer.state[parameter]=state
                        optimizer.param_groups[0]['lr']=max(1e-8,optimizer.param_groups[0]['lr']*.5)
                        rejected+=1
                    else:maxkl=max(maxkl,kl)
                    losses.append([float(policy.detach()),float(value_loss.detach()),float(imitation.detach())])
                if maxkl>.03:break
            completed=update
            if rejected==0 and maxkl<.01:
                optimizer.param_groups[0]['lr']=min(3e-6,optimizer.param_groups[0]['lr']*1.5)
            metric=dict(update=update,elapsed_s=time.monotonic()-started,iteration_s=time.monotonic()-tick,
                transitions=update*count,transitions_per_second=update*count/(time.monotonic()-rollout_started),
                physical_failures=failures,completed_training_tails=successes,mean_terminated_episode_controls=float(np.mean(age_sum)) if age_sum else None,
                motion_policy_fraction=float(torch.stack(active).mean()),
                mean_reward=float(buffers['reward'].mean()),root_error_mean_m=float(torch.stack(root).mean()),
                leg_rmse_mean_rad=float(torch.stack(legs).mean()),losses=np.mean(losses,axis=0).tolist(),kl=maxkl,
                rejected_actor_updates=rejected,actor_learning_rate=optimizer.param_groups[0]['lr'])
            log.write(json.dumps(metric)+'\n');log.flush()
            if update<=3 or update%10==0:print(json.dumps(metric),flush=True)
            if update in (args.updates//2,args.updates):save(f'{update:05d}',update)
            if (args.output/'STOP').exists() or time.monotonic()-started>args.wall_hours*3600:
                if update not in (args.updates//2,args.updates):save(f'{update:05d}',update)
                break
    write(args.output/'outcome.json',dict(completed_updates=completed,elapsed_s=time.monotonic()-started,
        simulation_qualified=False,hardware_authorized=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bank',type=Path,default=BASE/'causal_dynamics_v1/bank')
    p.add_argument('--initial',type=Path,default=BASE/'direct_target_causal_width512_student_v1/fit/student_head.pt')
    p.add_argument('--num-envs',type=int,default=128);p.add_argument('--steps',type=int,default=64)
    p.add_argument('--updates',type=int,default=800);p.add_argument('--bootstrap-updates',type=int,default=1000)
    p.add_argument('--wall-hours',type=float,default=3.)
    p.add_argument('--wait-initial-evaluation',action='store_true')
    p.add_argument('--balanced',action='store_true');p.add_argument('--resume',type=Path)
    args=p.parse_args()
    try:run(args)
    except BaseException as e:
        if args.output.exists():write(args.output/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
