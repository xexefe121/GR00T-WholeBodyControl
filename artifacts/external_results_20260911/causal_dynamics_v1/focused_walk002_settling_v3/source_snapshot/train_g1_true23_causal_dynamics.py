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


def validate_initialization(checkpoint,args,input_width,*,initial_only=False):
    metadata=checkpoint.get('request',{})
    if float(metadata.get('smooth_action_tau',0.))!=args.smooth_action_tau:
        raise ValueError('checkpoint action mapping differs from requested candidate')
    if bool(metadata.get('normalize_range_adapter',False))!=args.normalize_range_adapter:
        raise ValueError('checkpoint range-adapter normalization differs from requested candidate')
    if int(metadata.get('features',1323))!=input_width:
        raise ValueError('checkpoint observation width differs from environment')
    if bool(metadata.get('single_policy',False))!=args.single_policy:
        raise ValueError('checkpoint actor architecture differs from requested candidate')
    if bool(metadata.get('direct_body_goal',False))!=args.direct_body_goal:
        raise ValueError('checkpoint body goal semantics differ from requested candidate')
    if initial_only and not checkpoint.get('initialization_only',False):
        raise ValueError('--actor-init requires an explicitly initialization-only checkpoint')


def adapt_actor_learning_rates(groups,factor):
    for group in groups:
        group['lr']=min(group['max_lr'],max(group['min_lr'],group['lr']*factor))


def rollback_actor_update(actor,optimizer,old_state,old_optimizer,actor_groups):
    actor.load_state_dict(old_state)
    for parameter,state in old_optimizer.items():optimizer.state[parameter]=state
    adapt_actor_learning_rates(actor_groups,.5)


def gradient_norm(parameters):
    values=[p.grad.detach().norm() for p in parameters if p.grad is not None]
    return float(torch.stack(values).norm()) if values else 0.


def check_features(env):
    actual=env.observe().cpu().numpy()
    for i in range(min(8,env.count)):
        clip=int(env.clips[i]);frame=int(env.frames[i])
        refs={k:v[clip,:env.lengths[clip]].cpu().numpy() for k,v in env.references.items()}
        expected=features_numpy(env.q[i].cpu().numpy(),env.v[i].cpu().numpy(),refs,frame,
            env.default.cpu().numpy(),env.prior[i].cpu().numpy(),env.history_vector()[i].cpu().numpy())
        np.testing.assert_allclose(actual[i,:1323],expected,rtol=2e-5,atol=2e-5)
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
    env_class=CausalNative23Env
    if args.balanced:
        from gear_sonic.envs.mjlab.g1_true23_balanced_causal_dynamics import BalancedCausalNative23Env
        env_class=BalancedCausalNative23Env
    if args.single_policy:
        from gear_sonic.envs.mjlab.g1_true23_single_policy_dynamics import SinglePolicyNative23Env
        env_class=SinglePolicyNative23Env
    if args.direct_body_goal:
        from gear_sonic.envs.mjlab.g1_true23_direct_body_dynamics import DirectBodyNative23Env
        env_class=DirectBodyNative23Env
    env_options={}
    if args.focus_clip:
        from gear_sonic.envs.mjlab.g1_true23_focused_dynamics import FocusedDirectBodyNative23Env
        env_class=FocusedDirectBodyNative23Env;env_options['focus_clip']=args.focus_clip
        env_options['settle_objective']=args.settle_objective
    env=env_class(ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1',args.bank,args.num_envs,**env_options)
    input_width=int(env.observe().shape[-1])
    if args.single_policy:
        from gear_sonic.utils.g1_true23_single_policy import SinglePolicy
        if args.direct_body_goal:
            from gear_sonic.utils.g1_true23_direct_body_goal import DirectBodySinglePolicy
            SinglePolicy=DirectBodySinglePolicy
        weights=ROOT/'artifacts/g1_true23_six_hour_replan_20260910_v1/bfmzero_inference_v1/inference.safetensors'
        with np.load(ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/walk003/native_original.npz') as z:
            neutral={k:z[k][:1] for k in z.files if k!='fps'}
        actor=SinglePolicy(weights,env.c,neutral,env.expert_features[:,:1323].mean(0).cpu().numpy(),
            env.expert_features[:,:1323].std(0).cpu().numpy(),smooth_action_tau=args.smooth_action_tau,
            normalize_range_adapter=args.normalize_range_adapter).cuda()
    else:
        saved=torch.load(args.initial,map_location='cpu',weights_only=False)
        actor=Actor(saved,projected_mean=args.projected_mean).cuda()
    critic=nn.Sequential(nn.Linear(1323,256),nn.ELU(),nn.Linear(256,256),nn.ELU(),nn.Linear(256,1)).cuda()
    if args.actor_init is not None:
        initialization=torch.load(args.actor_init,map_location='cuda:0',weights_only=False)
        validate_initialization(initialization,args,input_width,initial_only=True)
        actor.load_state_dict(initialization['actor'])
    if args.resume is not None:
        resumed=torch.load(args.resume,map_location='cuda:0',weights_only=False)
        validate_initialization(resumed,args,input_width)
        actor.load_state_dict(resumed['actor']);critic.load_state_dict(resumed['critic'])
    if args.adapter_lr is None:
        actor_groups=[dict(name='actor',params=list(actor.parameters()),lr=args.actor_lr,min_lr=1e-8,max_lr=max(3e-6,args.actor_lr))]
    else:
        adapters=[];backbone=[]
        for name,parameter in actor.named_parameters():
            (adapters if name.startswith(('goal_adapter.','range_adapter.')) else backbone).append(parameter)
        if not adapters or not backbone:raise ValueError('differential learning rates require both adapters and pretrained actor')
        actor_groups=[dict(name='pretrained_actor',params=backbone,lr=args.actor_lr,min_lr=min(1e-8,args.actor_lr),max_lr=max(3e-6,args.actor_lr)),
            dict(name='goal_range_adapters',params=adapters,lr=args.adapter_lr,min_lr=min(1e-8,args.adapter_lr),max_lr=max(3e-6,args.adapter_lr))]
    optimizer=torch.optim.Adam([*actor_groups,dict(name='critic',params=list(critic.parameters()),lr=3e-4)])
    # Optimizer normalizes group dictionaries; retain its authoritative ones.
    actor_groups=optimizer.param_groups[:-1]
    env.training_extra_hold_controls=args.training_extra_hold_controls
    write(args.output/'input_checks.json',check_features(env))
    # Recondition the changed causal observations while preserving the initial
    # actor function. This avoids huge gradients on newly shifted features.
    if args.resume is None and not args.single_policy:
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
        actor_initialization_only=str(args.actor_init) if args.actor_init else None,fresh_critic=args.resume is None,
        actor_normalization_source='stored checkpoint buffers' if args.actor_init or args.resume else 'current expert bank',
        retained_neutral_balance=args.balanced,motion_takeover_seconds=.5 if args.balanced else 0,
        projected_exploration_mean=args.projected_mean,projection_gradient='straight_through' if args.projected_mean else None,
        training_extra_hold_controls=args.training_extra_hold_controls,evaluation_extra_hold_controls=1500,
        prior_updates=args.prior_updates,total_update_limit=args.prior_updates+args.updates,
        actor_width=512,features=input_width,future_reference_frames=0,reference_derivatives='backward',
        native_pd=True,physics_hz=500,control_hz=50,simulator='MJLab MuJoCo-Warp 3.5.0',
        final_referee='MuJoCo3.2.3',num_envs=args.num_envs,rollout_steps=args.steps,
        updates=args.updates,bootstrap_updates=args.bootstrap_updates,wall_hours=args.wall_hours,minibatches=args.minibatches,
        initial_actor_learning_rate=args.actor_lr,smooth_action_tau=args.smooth_action_tau,
        normalize_range_adapter=args.normalize_range_adapter,
        initial_adapter_learning_rate=args.adapter_lr,imitation_weight=args.imitation_weight,
        focus_clip=args.focus_clip,
        settle_objective=args.settle_objective,
        imitation_sampling='75pct chosen source,25pct chosen acquisition/return/standing' if args.focus_clip else 'uniform three training clips',
        optimizer_groups=[dict(name=g['name'],initial_lr=g['lr'],min_lr=g.get('min_lr'),max_lr=g.get('max_lr')) for g in optimizer.param_groups],
        separate_gradient_clipping=args.separate_gradient_clipping,
        milestones=[args.updates//2,args.updates],heldout='walk008',root_state='simulation_privileged',
        reset_schedule='50pct canonical lifecycle;50pct actual expert states;root XY velocity +/-0.03mps',
        hardware_authorized=False,simulation_qualified=False)
    if args.balanced:
        request['reset_schedule']='equal canonical starts, actual expert states, actual states near six interruption points; XY velocity +/-0.03mps'
    if args.single_policy:
        request.update(kind='single_trainable_bfm_causal_native23_dynamics_ppo',
            initial_checkpoint=str(weights),single_policy=True,bfm_actor_frozen=False,
            actor_width=None,actor_parameters=sum(p.numel() for p in actor.parameters()),
            retained_neutral_balance=False,motion_takeover_seconds=0,
            goal_conditioning='neutral feedback plus learned adapter from all1323 received-only full-body features',
            target_mapping='native interval sigmoid; zero range adapter preserves original balance within1e-6 native span',
            reset_schedule='equal canonical starts, actual expert states and six interruption states; XY velocity +/-0.03mps')
    if args.direct_body_goal:
        request.update(kind='direct_received_body_trainable_bfm_native23_ppo',direct_body_goal=True,
            goal_conditioning='current received native body pose through pretrained backward encoder; past-only goal blend; all1323 task/history features retained')
    if args.smooth_action_tau:
        request['target_mapping']='algebraic smooth native fraction; residual before saturation; ordinary derivatives; no changed physical bounds'
    if args.focus_clip:
        request['reset_schedule']='focused environment: actual source states, canonical starts and interruption recovery for '+args.focus_clip
        if args.settle_objective:
            request['reset_schedule']='10pct canonical,50pct actual source,40pct actual braking/interruption; XY velocity +/-0.03mps'
            request['settling_reward']='received-velocity quiet detector; dense root/joint velocity, tilt, pose and yaw costs; completion bonus requires settled state'
    write(args.output/'request.json',request)
    print(json.dumps(dict(ready=True,**request)),flush=True)

    def save(label,update):
        path=args.output/f'actor_{label}.pt'
        checkpoint=dict(actor=actor.state_dict(),critic=critic.state_dict(),optimizer=optimizer.state_dict(),
            source=request['initial_checkpoint'],request=request,update=update,global_update=args.prior_updates+update,elapsed_s=time.monotonic()-started)
        torch.save(checkpoint,path)
        cpu=copy.deepcopy(actor).cpu().eval()
        torch.onnx.export(cpu,torch.zeros(1,input_width),str(path.with_suffix('.onnx')),input_names=['features'],
            output_names=['normalized_target'],dynamic_axes={'features':{0:'batch'},'normalized_target':{0:'batch'}},
            opset_version=17,dynamo=False)
        np.savez(path.with_suffix('.normalization.npz'),span=cpu.span.numpy(),default=cpu.default.numpy(),limits=cpu.limits.numpy())
        print(json.dumps(dict(checkpoint=str(path),update=update)),flush=True)

    save('initial',0)
    # Short adaptation to changed observation semantics, then physics drives
    # optimization. Clips are sampled uniformly, not dominated by long Pico.
    perclip=[(env.expert_resets[:,0]==i).nonzero().flatten() for i in range(3)]
    focused_auxiliary=None
    if args.focus_clip:
        focus_index=next(i for i,row in enumerate(env.meta['clips']) if row['name']==args.focus_clip)
        row=env.meta['clips'][focus_index]
        source=((env.expert_resets[:,0]==focus_index)&(env.expert_resets[:,1]>=row['source_start']+11)
            &(env.expert_resets[:,1]<row['source_stop']+11)).nonzero().flatten()
        if not len(source):raise ValueError('focused clip has no actual expert source states')
        perclip=[source]
        focused_auxiliary=((env.expert_resets[:,0]==focus_index)&((env.expert_resets[:,1]<row['source_start']+11)
            |(env.expert_resets[:,1]>=row['source_stop']+11))).nonzero().flatten()
        if not len(focused_auxiliary):raise ValueError('focused clip has no actual expert acquisition, return or standing states')
    boot=torch.optim.Adam(actor.parameters(),lr=3e-5)
    def expert_batch(n=256):
        if focused_auxiliary is not None:
            source_count=int(.75*n)
            ids=torch.cat((perclip[0][torch.randint(len(perclip[0]),(source_count,),device='cuda')],
                focused_auxiliary[torch.randint(len(focused_auxiliary),(n-source_count,),device='cuda')]))
        else:
            ids=torch.cat([idx[torch.randint(len(idx),(n//len(perclip)+1,),device='cuda')] for idx in perclip])[:n]
        return env.expert_features[ids],(env.expert_targets[ids]-actor.default)/actor.span
    for i in range(args.bootstrap_updates):
        x,y=expert_batch()
        pred=actor(x);loss=((pred-y)*actor.span).square().mean()
        boot.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(actor.parameters(),1);boot.step()
        if i%200==0: print(json.dumps(dict(bootstrap=i,rmse_rad=float(loss.detach().sqrt()),elapsed_s=time.monotonic()-started)),flush=True)
    save('bootstrap',0)
    probe_ids=torch.cat([idx[torch.linspace(0,len(idx)-1,min(32,len(idx)),device='cuda').long()] for idx in perclip])
    probe_features=env.expert_features[probe_ids];probe_teacher=env.expert_targets[probe_ids]
    with torch.no_grad():
        probe_initial=actor.target(actor(probe_features)).clone();probe_previous=probe_initial.clone()
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
        obs=(args.steps,args.num_envs,input_width),action=(args.steps,args.num_envs,23),
        mean=(args.steps,args.num_envs,23),logprob=(args.steps,args.num_envs),value=(args.steps,args.num_envs),
        reward=(args.steps,args.num_envs),done=(args.steps,args.num_envs),policy_weight=(args.steps,args.num_envs)).items()}
    rollout_started=time.monotonic();completed=0
    with (args.output/'metrics.jsonl').open('x') as log:
        for update in range(1,args.updates+1):
            tick=time.monotonic();failures=successes=0;age_sum=[];root=[];legs=[];active=[];source_motion=[];tracking_failures=0;settled_completions=0
            with torch.no_grad():
                for t in range(args.steps):
                    mean=actor(obs)
                    distribution=torch.distributions.Normal(mean,std)
                    action=distribution.sample()
                    buffers['obs'][t]=obs;buffers['action'][t]=action;buffers['mean'][t]=mean
                    buffers['logprob'][t]=distribution.log_prob(action).sum(-1)
                    buffers['value'][t]=critic((obs[:,:1323]-actor.mean)/actor.scale).squeeze(-1)
                    obs,reward,done,info=env.step(actor.target(action))
                    buffers['reward'][t]=reward;buffers['done'][t]=done
                    weight=(info['motion_fraction']>0).float() if 'motion_fraction' in info else torch.ones_like(reward)
                    buffers['policy_weight'][t]=weight;active.append(weight.mean())
                    failures+=int(info['failed'].sum());successes+=int(info['complete'].sum())
                    settled_completions+=int(info['rewarded_complete'].sum())
                    if 'source_motion' in info:source_motion.append(info['source_motion'].float().mean())
                    if 'tracking_failed' in info:tracking_failures+=int(info['tracking_failed'].sum())
                    if done.any():age_sum.extend(info['age'][done].cpu().tolist())
                    root.append(info['root_error'].mean());legs.append(info['leg_error'].mean())
                next_value=critic((obs[:,:1323]-actor.mean)/actor.scale).squeeze(-1)
                advantage=torch.zeros_like(buffers['reward']);carry=torch.zeros_like(next_value)
                for t in range(args.steps-1,-1,-1):
                    alive=1-buffers['done'][t]
                    delta=buffers['reward'][t]+gamma*next_value*alive-buffers['value'][t]
                    carry=delta+gamma*lam*alive*carry;advantage[t]=carry;next_value=buffers['value'][t]
                returns=(advantage+buffers['value']).flatten()
                adv=advantage.flatten();adv=(adv-adv.mean())/(adv.std()+1e-8)
            flat={k:v.flatten(0,1) for k,v in buffers.items()}
            losses=[];maxkl=0.;rejected=0
            group_gradients={group['name']:[] for group in optimizer.param_groups}
            for epoch in range(4):
                for ix in torch.randperm(count,device='cuda').chunk(args.minibatches):
                    x=flat['obs'][ix];mean=actor(x)
                    dist=torch.distributions.Normal(mean,std)
                    lp=dist.log_prob(flat['action'][ix]).sum(-1)
                    delta=lp-flat['logprob'][ix];ratio=delta.exp()
                    weight=flat['policy_weight'][ix]
                    active_batch=bool(weight.any())
                    policy=-(torch.minimum(ratio*adv[ix],ratio.clamp(.8,1.2)*adv[ix])*weight).sum()/weight.sum().clamp_min(1)
                    value=critic((x[:,:1323]-actor.mean)/actor.scale).squeeze(-1)
                    value_loss=(value-returns[ix]).square().mean()
                    bx,by=expert_batch(128)
                    imitation=((actor(bx)-by)*actor.span).square().mean()
                    loss=policy+.5*value_loss+args.imitation_weight*imitation
                    if not torch.isfinite(loss):raise RuntimeError('nonfinite PPO loss')
                    old_state=copy.deepcopy(actor.state_dict())
                    old_optimizer={p:copy.deepcopy(optimizer.state.get(p,{})) for p in actor.parameters()}
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    if not active_batch:
                        for parameter in actor.parameters():parameter.grad=None
                    for group in optimizer.param_groups:
                        group_gradients[group['name']].append(gradient_norm(group['params']))
                    if args.separate_gradient_clipping:
                        nn.utils.clip_grad_norm_(actor.parameters(),.5)
                        nn.utils.clip_grad_norm_(critic.parameters(),.5)
                    else:
                        nn.utils.clip_grad_norm_(list(actor.parameters())+list(critic.parameters()),.5)
                    optimizer.step()
                    with torch.no_grad():
                        per_state_kl=.5*((actor(x)-flat['mean'][ix])/std).square().sum(-1)
                        kl=float((per_state_kl*weight).sum()/weight.sum().clamp_min(1))
                    if not np.isfinite(kl) or kl>.03:
                        rollback_actor_update(actor,optimizer,old_state,old_optimizer,actor_groups)
                        rejected+=1
                    else:maxkl=max(maxkl,kl)
                    losses.append([float(policy.detach()),float(value_loss.detach()),float(imitation.detach())])
                if maxkl>.03:break
            completed=update
            if rejected==0 and maxkl<.01:
                adapt_actor_learning_rates(actor_groups,1.5)
            with torch.no_grad():
                probe_targets=actor.target(actor(probe_features))
                probe_movement=probe_targets-probe_initial;probe_update=probe_targets-probe_previous
                probe_error=probe_targets-probe_teacher
                movement=dict(states=len(probe_ids),from_initial_rms_rad=float(probe_movement.square().mean().sqrt()),
                    from_initial_max_rad=float(probe_movement.abs().max()),update_rms_rad=float(probe_update.square().mean().sqrt()),
                    teacher_rmse_rad=float(probe_error.square().mean().sqrt()),
                    teacher_leg_rmse_rad=float(probe_error[:,:12].square().mean().sqrt()))
                probe_previous.copy_(probe_targets)
            metric=dict(update=update,global_update=args.prior_updates+update,elapsed_s=time.monotonic()-started,iteration_s=time.monotonic()-tick,
                transitions=update*count,transitions_per_second=update*count/(time.monotonic()-rollout_started),
                physical_failures=failures,completed_training_tails=successes,mean_terminated_episode_controls=float(np.mean(age_sum)) if age_sum else None,
                rewarded_training_completions=settled_completions,
                motion_policy_fraction=float(torch.stack(active).mean()),
                source_motion_fraction=float(torch.stack(source_motion).mean()) if source_motion else None,
                tracking_failures=tracking_failures,
                mean_reward=float(buffers['reward'].mean()),root_error_mean_m=float(torch.stack(root).mean()),
                leg_rmse_mean_rad=float(torch.stack(legs).mean()),losses=np.mean(losses,axis=0).tolist(),kl=maxkl,
                rejected_actor_updates=rejected,actor_learning_rate=optimizer.param_groups[0]['lr'],
                group_learning_rates={group['name']:group['lr'] for group in optimizer.param_groups},
                group_preclip_gradient_norms={name:dict(mean=float(np.mean(values)),maximum=float(np.max(values))) for name,values in group_gradients.items()},
                fixed_expert_probe=movement)
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
    p.add_argument('--actor-init',type=Path)
    p.add_argument('--projected-mean',action='store_true')
    p.add_argument('--training-extra-hold-controls',type=int,default=1500)
    p.add_argument('--prior-updates',type=int,default=0)
    p.add_argument('--single-policy',action='store_true');p.add_argument('--minibatches',type=int,default=4)
    p.add_argument('--actor-lr',type=float,default=3e-6)
    p.add_argument('--adapter-lr',type=float)
    p.add_argument('--imitation-weight',type=float,default=.2)
    p.add_argument('--direct-body-goal',action='store_true')
    p.add_argument('--focus-clip',choices=('walk002','walk003','pico'))
    p.add_argument('--settle-objective',action='store_true')
    p.add_argument('--smooth-action-tau',type=float,default=0.)
    p.add_argument('--normalize-range-adapter',action='store_true')
    p.add_argument('--separate-gradient-clipping',action='store_true')
    args=p.parse_args()
    if args.focus_clip and not (args.single_policy and args.direct_body_goal):
        raise ValueError('--focus-clip requires --single-policy and --direct-body-goal')
    if args.settle_objective and not args.focus_clip:raise ValueError('--settle-objective requires --focus-clip')
    if args.actor_init is not None and (args.resume is not None or not args.single_policy):
        raise ValueError('--actor-init requires --single-policy and cannot be combined with --resume')
    if args.adapter_lr is not None and (not args.single_policy or not np.isfinite(args.adapter_lr) or args.adapter_lr<=0):
        raise ValueError('--adapter-lr requires --single-policy and a positive finite rate')
    if not np.isfinite(args.actor_lr) or args.actor_lr<=0:raise ValueError('actor learning rate must be positive and finite')
    if not np.isfinite(args.imitation_weight) or args.imitation_weight<0:raise ValueError('imitation weight must be finite and nonnegative')
    if not 0<=args.smooth_action_tau<=.1:raise ValueError('smooth action tau must be in [0,.1]')
    if args.smooth_action_tau and not args.single_policy:raise ValueError('smooth action map requires --single-policy')
    if args.normalize_range_adapter and not args.single_policy:raise ValueError('range adapter normalization requires --single-policy')
    if args.direct_body_goal and not args.single_policy:raise ValueError('direct body goal requires --single-policy')
    if args.single_policy and (args.balanced or args.projected_mean or args.bootstrap_updates):
        raise ValueError('single policy uses no supervisor, STE or supervised bootstrap; pass --bootstrap-updates 0')
    if args.training_extra_hold_controls<0 or args.prior_updates<0:raise ValueError('control and update counts cannot be negative')
    try:run(args)
    except BaseException as e:
        if args.output.exists():write(args.output/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
