"""Bounded factory-initialized native23 PPO with native full-motion decisions.

Runs locally on the existing MJLab/CUDA environment. No scheduler, paid service,
robot network, or hardware publisher. Checkpoint evaluation invokes the native
Windows MuJoCo 3.2.3 referee on all four complete recordings plus 30s hold.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
import torch
from torch import nn
import yaml
from gear_sonic.envs.mjlab.g1_true23_factory_dynamics import FactoryNative23Env
from gear_sonic.utils.g1_true23_factory_policy import FactoryReceivedActor,WIDTH
from gear_sonic.scripts.train_g1_true23_causal_dynamics import generalized_advantages

ROOT=Path(__file__).resolve().parents[2]


def write(path,data):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def windows_path(path):
    return subprocess.check_output(['wslpath','-w',str(path.resolve())],text=True).strip()


def evaluate(actor_path,output,*,native=False,locomotion=False,task_commands=False,native_targets=False):
    output.mkdir(parents=True,exist_ok=False)
    reports={}
    python='/mnt/c/Users/camer/AppData/Local/Programs/Python/Python310/python.exe'
    script=windows_path(ROOT/'artifacts/onboard_inspection_20260912/run_factory_pose_sim.py')
    for clip in ('walk002','walk003','pico','walk008'):
        folder=output/clip
        command=[python,script,'--clip',clip,'--actor',windows_path(actor_path),'--output',windows_path(folder)]
        if native:command+=['--native-conditioned']
        if task_commands:command+=['--task-commands']
        elif locomotion:command+=['--locomotion-conditioned']
        if native_targets:command+=['--native-targets']
        result=subprocess.run(command,
            capture_output=True,text=True,timeout=240)
        (output/f'{clip}.stdout.log').write_text(result.stdout)
        (output/f'{clip}.stderr.log').write_text(result.stderr)
        if result.returncode:raise RuntimeError(f'native evaluation {clip} exited {result.returncode}: {result.stderr[-1500:]}')
        reports[clip]=json.loads((folder/'report.json').read_text())
    summary={'clips':reports,'all_passed':all(r['passed'] for r in reports.values()),
        'physical_completions':sum(r['physical_complete'] for r in reports.values()),
        'physical_seconds':sum(r['simulation_seconds'] for r in reports.values()),
        'future_reference_frames':0,'extra_standing_seconds':30,'independent_clock_tested':False,
        'simulation_ready':False}
    write(output/'report.json',summary)
    return summary


def run(args):
    if args.command_space_exploration and not args.task_commands:
        raise ValueError('Command-space exploration requires --task-commands')
    if args.full_body_corrections and not (args.task_commands and args.command_space_exploration):
        raise ValueError('Full-body corrections require task commands and command-space exploration')
    if args.native_targets and not (args.full_body_corrections and args.native_physics):
        raise ValueError('Native targets require full-body corrections and original native physics')
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic(); deadline=started+args.wall_minutes*60
    device=args.policy_device
    torch.set_num_threads(4 if device=='cpu' else 1);torch.manual_seed(getattr(args,'seed',20260912))
    torch.backends.cuda.matmul.allow_tf32=False
    cfg_path=args.firmware/'decoded_configs/policies/cpy_dance/dance.yaml'
    if args.native_base:cfg_path=args.firmware/'decoded_configs/policies/mimic_test/fsm_mimic_test.yaml'
    if args.locomotion_base:cfg_path=args.firmware/'decoded_configs/policies/human_loco/fsm_human_loco_config.yaml'
    cfg=yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    source=args.firmware/'ai_sport_files/ai_sport_8.4.2.222/module/ai_sport/file/unitree/module/ai_sport/policies/cpy_dance/Feb13_20-31-05_/actor.onnx'
    if args.locomotion_base:
        from gear_sonic.envs.mjlab.g1_true23_locomotion_conditioned_dynamics import LocomotionConditionedEnv
        from gear_sonic.utils.g1_true23_locomotion_conditioned import LocomotionConditionedActor,LOCOMOTION_CONDITIONED_WIDTH
        env_class=LocomotionConditionedEnv
        if args.task_commands:
            from gear_sonic.envs.mjlab.g1_true23_task_command_dynamics import TaskCommandEnv
            from gear_sonic.utils.g1_true23_task_commands import TaskCommandActor,TASK_COMMAND_WIDTH
            env_class=TaskCommandEnv
            if args.native_targets:
                from gear_sonic.envs.mjlab.g1_true23_native_target_dynamics import NativeTargetEnv
                env_class=NativeTargetEnv
        source=args.firmware/'human_loco_trainable_v1/factory_weights.npz'
        additional={}
        if getattr(args,'controller_memory_bank',None):
            from gear_sonic.envs.mjlab.g1_true23_matched_memory_dynamics import MatchedMemoryNative23Env
            env_class=MatchedMemoryNative23Env
            additional=dict(memory_bank=args.controller_memory_bank,reset_memory=args.reset_memory,
                preview_library=args.preview_library,experiment_seed=args.seed)
        env=env_class(ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1',args.bank,cfg_path,count=args.num_envs,device=device,
            command_delay_substeps=args.command_delay_substeps,canonical_starts=args.canonical_starts,canonical_worlds=args.canonical_worlds,
            physics_backend='mjbatch' if args.native_physics else 'warp',**additional)
        if additional:torch.manual_seed(args.seed)
        if args.task_commands:
            if args.native_targets:
                from gear_sonic.utils.g1_true23_native_targets import NativeTargetActor
                actor=NativeTargetActor(source,cfg_path,env.c,args.task_base_checkpoint).to(device)
            else:
                actor=TaskCommandActor(source,cfg_path,env.c['joint_limits'],args.task_base_checkpoint,
                    full_body_corrections=args.full_body_corrections).to(device)
            width=TASK_COMMAND_WIDTH
        else:
            actor=LocomotionConditionedActor(source,cfg_path,env.c['joint_limits'],neutral_recovery_gate=args.neutral_recovery_gate).to(device);width=LOCOMOTION_CONDITIONED_WIDTH
    elif args.native_base:
        from gear_sonic.envs.mjlab.g1_true23_factory_conditioned_dynamics import NativeFactoryConditionedEnv
        from gear_sonic.utils.g1_true23_factory_conditioned import NativeFactoryConditionedActor,NATIVE_CONDITIONED_WIDTH
        source=args.firmware/'native23_trainable_v1/factory_weights.npz'
        env=NativeFactoryConditionedEnv(ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1',args.bank,cfg_path,count=args.num_envs)
        actor=NativeFactoryConditionedActor(source,cfg_path,env.c['joint_limits']).cuda();width=NATIVE_CONDITIONED_WIDTH
    else:
        env=FactoryNative23Env(ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1',args.bank,cfg_path,count=args.num_envs)
        actor=FactoryReceivedActor(source,cfg['default_dof_pos'],env.c['joint_limits']).cuda();width=WIDTH
    critic=nn.Sequential(nn.Linear(width,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,1)).to(device)
    motion_prior=None
    if args.motion_prior:
        from gear_sonic.utils.g1_true23_motion_prior import MotionPrior
        motion_prior=MotionPrior(args.motion_prior,device)
        env.capture_motion_prior=True
    env.terminate_tracking_errors=not args.continue_tracking_errors
    def evaluate_candidate(path,folder):
        callback=getattr(args,'evaluation_callback',None)
        if callback is not None:return callback(actor,path,folder,env)
        return evaluate(path,folder,native=args.native_base,locomotion=args.locomotion_base,
            task_commands=args.task_commands,native_targets=args.native_targets)
    request={'kind':'factory_received_full_range_native23_ppo','source':str(source),'bank':str(args.bank),
        'all_factory_weights_trainable':True,'added_input':'measured root/feet/hand/head errors and velocity; zero initial weights',
        'features':width,'future_reference_frames':0,'physics_hz':500,'control_hz':50,'native_limit_brake':True,
        'target_margin_rad':.06,'state_projection':False,'paid_compute':False,'hardware_commands':False,
        'physics_backend':'native MuJoCo3.2.3 mjbatch' if args.native_physics else 'MuJoCo Warp',
        'policy_training_device':device,
        'reset_roles':'75% motion across three training clips;12.5% acquisition/braking/input loss;12.5% actual quiet states',
        'standing_example_clips':[env.meta['clips'][i]['name'] for i in env.standing_clips],
        'held_out':'walk008','num_envs':args.num_envs,'rollout_steps':args.steps,'updates':args.updates,
        'checkpoint_updates':args.checkpoint_updates or [args.updates//2,args.updates],'wall_minutes':args.wall_minutes,
        'prior_pilot_updates':args.prior_updates,'total_pilot_update_limit':args.prior_updates+args.updates,
        'initial_standing_examples':int(len(env.initial_standing_pool)),
        'standing_reset_mix':'half initial actual quiet states,half terminal actual quiet states',
        'policy_update_guard':'analytic KL <=0.02 after each accepted minibatch; rollback model and Adam on rejection',
        'bootstrap_updates':args.bootstrap_updates,'teacher_targets':'PD torque-equivalent under factory gains plus limit brake',
        'bootstrap_learning_rate':args.bootstrap_lr,'command_delay_substeps':args.command_delay_substeps,
        'neutral_recovery_gate':args.neutral_recovery_gate,
        'canonical_episode_fraction':args.canonical_worlds/8 if args.canonical_starts else 0.,
        'imitation_weight':args.imitation_weight,
        'balanced_canonical_recordings':args.canonical_starts,
        'tracking_error_terminates_training_episode':not args.continue_tracking_errors,
        'surviving_reward_transform':'softplus of unchanged dense task reward' if args.continue_tracking_errors else None,
        'quiet_reference_test':'root, feet, hand/head linear speeds<0.02m/s; joint, root and hand/head angular speeds<0.05rad/s',
        'imitation_state_target_pairing':'expert rows only; overwritten canonical-start states excluded',
        'resume_actor':str(args.resume_actor) if args.resume_actor else None,
        'evaluation':'native MuJoCo3.2.3; all four complete original-speed motions plus30s; no rollout resets',
        'selection':'experimental only; no live promotion without all goal gates','automatic_extension':False}
    if args.native_base:
        request.update(kind='preserved_native23_factory_full_range_conditioning_ppo',all_factory_weights_trainable=False,
            frozen_backbone='exact verified native23 factory observation memory and balance actor; no missing-joint transfer',
            added_input='all1323 current/past received full-body, task, robot and command-history features plus causal geometric closure',
            output='full native interval per joint; no small residual amplitude restriction',
            neutral_legs='preserved only for received neutral closure with root error<3cm and yaw<0.06rad; arms stay trainable',
            standing_arms_tracking='remains part of full-body rewards and native evaluation')
    if args.locomotion_base:
        request.update(kind='native_locomotion_received_full_body_ppo',all_factory_weights_trainable=False,
            frozen_backbone='verified native12-leg factory locomotion with actual applied-action memory',
            added_input='current/past received joints, feet, hands, head, root and robot state; causal geometric closure',
            output='12 full-range leg corrections plus 11 deterministic received upper-body q/dq targets',
            exploration='12 leg dimensions only; upper-body targets remain deterministic',
            source_episode_seconds=env.source_duration*.02,
            standing_arms_tracking='all original body tracking rewards and native evaluation retained')
    if args.canonical_starts:
        request['reset_roles']=f'{(6-args.canonical_worlds)*12.5:g}% expert motion;{args.canonical_worlds*12.5:g}% complete actual-start episodes;12.5% transitions/input loss;12.5% actual quiet states'
    if args.task_commands:
        request.update(kind='native23_received_task_command_ppo',task_base_checkpoint=str(args.task_base_checkpoint),
            added_input='measured12 leg position/velocity errors, two foot and three hand/head relative-position errors',
            trainable_commands='two continuous foot phase offsets and three factory velocity corrections, plus12 full-range joint corrections',
            retained_baseline='frozen native-trained Pico controller; alpha0.9 from actual applied targets',
            decoupled_critic=args.decoupled_critic,critic_warmup_updates=args.critic_warmup_updates)
        if args.command_space_exploration:
            request.update(kind='native23_received_command_space_ppo',
                exploration='17 latent commands:12 full-range leg corrections,two coherent foot phases,three factory velocities',
                exploration_std_near_zero=dict(joint_target_rad=.03,foot_phase_rad=.3,velocity_mps_mps_radps=[.05,.05,.2]),
                policy_likelihood='Gaussian over sampled latent commands, before frozen balance-network mapping',
                runtime_export='same deterministic1582-input,23-target actor; no stochastic runtime commands')
        if args.full_body_corrections:
            request.update(kind='native23_received_full_body_command_space_ppo',full_body_corrections=True,
                output='23 full-range joint corrections around retained Pico legs and received upper-body q/dq',
                trainable_commands='23 joint corrections, two continuous foot phase offsets, three factory velocity corrections',
                exploration='28 latent commands; every native joint receives independent 0.03rad target exploration',
                standing_arms_tracking='waist and arms trainable; all original full-body objectives and limits retained')
    if motion_prior is not None:
        import hashlib
        request.update(motion_prior=dict(dataset=str(args.motion_prior),
            sha256=hashlib.sha256(args.motion_prior.read_bytes()).hexdigest(),
            physical_training_clips=motion_prior.clip_names,transition_features=142,
            weight=args.motion_prior_weight,discriminator_steps_per_rollout=8,
            discriminator_learning_rate=1e-4,gradient_penalty_coefficient=5,
            replay_capacity=50000,balanced_recording_sampling=True,
            original_task_rewards_retained=True,failed_transition_style_reward=0,
            actor_observation_unchanged=True,runtime_discriminator=False,
            method='https://xbpeng.github.io/projects/AMP/AMP_2021.pdf'))
    if args.native_targets:
        request.update(kind='native23_received_original_actuation_ppo',native_targets=True,
            target_margin_rad=0.,native_limit_brake=False,
            actuation='original native benchmark PD, effort caps and full legal target interval',
            factory_prior='boundary torque conversion only; new corrections use native target units',
            teacher_targets='original recorded native PD targets; no gain conversion or target-margin truncation',
            factory_action_memory='instantaneous factory-gain equivalent of actual native targets',
            received_action_history='actual applied native targets',original_physical_limits=True)
    if getattr(args,'controller_memory_bank',None):
        request.update(controller_snapshot_version=env.memory_manifest['snapshot_version'],wrapper=env.memory_manifest['wrapper'],
            reset_memory=args.reset_memory,controller_memory_sha256=env.memory_manifest['memory_sha256'],seed=args.seed,
            native_preview_threads=dict(outer=8,inner=1),command_delay_substeps=6,
            paired_physical_resets='independent per-world ordinal RNG, same seed in both arms',
            evaluation='fixed native physical segments with replayed controller memory for both learned arms')
        if getattr(args,'runtime_wrapper',None):
            complete_wrapper=json.loads(Path(args.runtime_wrapper).read_text())
            if any(complete_wrapper.get(k)!=v for k,v in request['wrapper'].items()):
                raise ValueError('Pinned runtime wrapper disagrees with reconstructed memory')
            request['wrapper']=complete_wrapper
        request.update(native_solver_memory_preserved_across_unrelated_resets=True,
                       native_solver_memory_preserved_during_scoring=True)
    write(args.output/'request.json',request)

    value_mean=value_scale=None
    training_state={}
    def save(update,label=None):
        stem=f'actor_{label or f"{update:05d}"}'
        path=args.output/(stem+'.pt')
        payload={'actor':actor.state_dict(),'critic':critic.state_dict(),'update':update,'request':request,
            'critic_mean':value_mean,'critic_scale':value_scale}
        if training_state:
            payload.update(optimizer=training_state['optimizer'].state_dict(),
                critic_optimizer=None if training_state['critic_optimizer'] is None else training_state['critic_optimizer'].state_dict(),
                torch_rng=torch.get_rng_state())
            if hasattr(env,'reset_rng'):
                payload.update(reset_rng=[copy.deepcopy(r.bit_generator.state) for r in env.reset_rng],
                    delay_rng=copy.deepcopy(env.delay_rng.bit_generator.state))
        if motion_prior is not None:
            payload.update(motion_prior=motion_prior.state_dict(),motion_prior_optimizer=motion_prior.optimizer.state_dict())
        torch.save(payload,path)
        cpu=copy.deepcopy(actor).cpu().eval()
        onnx_path=path.with_suffix('.onnx')
        torch.onnx.export(cpu,torch.zeros(1,width),str(onnx_path),input_names=['features'],output_names=['normalized_target'],
            opset_version=17,dynamo=False,dynamic_axes={'features':{0:'batch'},'normalized_target':{0:'batch'}})
        if 'wrapper' in request:write(path.with_suffix('.wrapper.json'),request['wrapper'])
        return onnx_path

    initial=save(0,'factory')
    # Bounded initialization on actual expert physics states. Recondition the
    # factory normalizers without changing its function before any optimizer.
    xs,ys=[],[]
    for _ in range(64):
        env.reset(torch.arange(env.count,device=device))
        valid=getattr(env,'reset_teacher_valid',torch.ones(env.count,device=device,dtype=torch.bool))
        xs.append(env.observe()[valid].clone());ys.append(env.reset_teacher[valid].clone())
    expert_x,expert_y=torch.cat(xs),torch.cat(ys)
    with torch.no_grad():
        sample=expert_x[:256]
        if args.locomotion_base:
            before=actor(sample).clone()
            head=actor.head_input(expert_x)[0]
            actor.goal_mean.copy_(head.mean(0));actor.goal_scale.copy_(head.std(0).clamp_min(.1))
            change=float((actor(sample)-before).abs().max())
        elif args.native_base:
            before=actor(sample).clone()
            actor.goal_mean.copy_(expert_x[:,380:1703].mean(0));actor.goal_scale.copy_(expert_x[:,380:1703].std(0).clamp_min(.1))
            change=float((actor(sample)-before).abs().max())
        else:
            before=actor.raw29(sample).clone()
            encoded=actor.memory((expert_x[:,:465]-actor.memory_mean)/actor.memory_scale)
            velocity=actor.estimator(encoded)
            joined=torch.cat((velocity,encoded,expert_x[:,465:633]),-1)
            mean=joined.mean(0,keepdim=True);scale=joined.std(0,keepdim=True).clamp_min(.1)
            layer=actor.actor[0];weight=layer.weight.clone()
            layer.bias.add_((weight@((mean-actor.actor_mean)/actor.actor_scale).T).squeeze(-1))
            layer.weight.mul_(scale/actor.actor_scale)
            actor.actor_mean.copy_(mean);actor.actor_scale.copy_(scale)
            change=float((actor.raw29(sample)-before).abs().max())
        if change>2e-4:raise ValueError(f'factory normalization reparameterization changed output: {change}')
    if args.resume_actor:
        if args.bootstrap_updates:raise ValueError('Resuming an actor requires --bootstrap-updates0')
        saved=torch.load(args.resume_actor,map_location=device,weights_only=False)
        if bool(saved['request'].get('neutral_recovery_gate',False))!=args.neutral_recovery_gate:
            raise ValueError('resumed actor must retain its neutral recovery gate setting')
        if bool(saved['request'].get('full_body_corrections',False))!=args.full_body_corrections:
            raise ValueError('resumed actor must retain its joint command dimensions')
        if bool(saved['request'].get('native_targets',False))!=args.native_targets:
            raise ValueError('resumed actor must retain its actuator interface')
        actor.load_state_dict(saved['actor'],strict=True)
        if getattr(args,'controller_memory_bank',None):
            if saved.get('critic_mean') is None or saved.get('critic_scale') is None:
                raise ValueError('Matched comparison requires saved identical critic normalization')
            critic.load_state_dict(saved['critic'],strict=True)
        if motion_prior is not None and 'motion_prior' in saved:
            if saved['request']['motion_prior']['sha256']!=request['motion_prior']['sha256']:
                raise ValueError('Resumed motion prior must retain its demonstration dataset')
            motion_prior.load_state_dict(saved['motion_prior'],strict=True)
            motion_prior.optimizer.load_state_dict(saved['motion_prior_optimizer'])
        # Normalization buffers travel with the resumed actor. Critic is fresh
        # because earlier checkpoints did not store its input normalizer.
    boot=torch.optim.Adam(actor.parameters(),lr=args.bootstrap_lr)
    action_dimensions=12 if args.locomotion_base and not args.full_body_corrections else 23
    for update in range(args.bootstrap_updates):
        indices=torch.randint(len(expert_x),(512,),device=device)
        target=actor.target(actor(expert_x[indices]))
        loss=((target[:,:action_dimensions]-expert_y[indices,:action_dimensions])/.25).square().mean()
        boot.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(actor.parameters(),1);boot.step()
        if update%50==0:print(json.dumps({'bootstrap':update,'target_rmse_rad':float(loss.detach().sqrt()*.25),'elapsed_seconds':time.monotonic()-started}),flush=True)
    baseline=save(0,'bootstrap')
    write(args.output/'running.json',{'status':'evaluating_bootstrap','elapsed_seconds':time.monotonic()-started})
    baseline_report=evaluate_candidate(baseline,args.output/'eval_bootstrap')
    print(json.dumps({'native_baseline_completions':baseline_report['physical_completions'],'seconds':baseline_report['physical_seconds']}),flush=True)
    if args.native_base or args.locomotion_base:
        groups=[{'params':actor.goal_head.parameters(),'lr':3e-5,'name':'full_range_goal_head'}]
    else:
        groups=[{'params':[p for n,p in actor.named_parameters() if not n.startswith('error_input.')],'lr':1e-5,'name':'factory'},
            {'params':actor.error_input.parameters(),'lr':3e-5,'name':'body_error'}]
    optimizer=torch.optim.Adam([*groups,{'params':critic.parameters(),'lr':3e-4,'name':'critic'}])
    critic_optimizer=torch.optim.Adam(critic.parameters(),lr=3e-4) if args.decoupled_critic else None
    training_state.update(optimizer=optimizer,critic_optimizer=critic_optimizer)
    if getattr(args,'resume_training',False):
        if saved['request'].get('wrapper')!=request.get('wrapper') or saved['request'].get('reset_memory')!=args.reset_memory:
            raise ValueError('Training continuation changed wrapper or memory arm')
        optimizer.load_state_dict(saved['optimizer'])
        if critic_optimizer is not None:critic_optimizer.load_state_dict(saved['critic_optimizer'])
        torch.set_rng_state(saved['torch_rng'])
        for rng,state in zip(env.reset_rng,saved['reset_rng']):rng.bit_generator.state=state
        env.delay_rng.bit_generator.state=saved['delay_rng']
    # Fixed critic scaling; keeps large absolute world coordinates and joint
    # units from determining value gradients. Actor retains factory scaling.
    value_mean=expert_x.mean(0);value_scale=expert_x.std(0).clamp_min(.1)
    if getattr(args,'controller_memory_bank',None):
        value_mean=saved['critic_mean'].to(device);value_scale=saved['critic_scale'].to(device)
    def value(x):return critic(((x-value_mean)/value_scale).clamp(-20,20)).squeeze(-1)
    env.reset(torch.arange(env.count,device=device))
    noise=torch.full((env.count,action_dimensions),.03,device=device)/actor.span[:action_dimensions]
    if args.command_space_exploration:
        n=actor.joint_command_count
        joint_sigma=.03/((actor.limits[:n,1]-actor.limits[:n,0])*.5)
        sigma=torch.cat((joint_sigma,joint_sigma.new_full((2,),.3/np.pi),
            joint_sigma.new_tensor([.05/.6,.05/.5,.2/3.])))
        noise=sigma[None].expand(env.count,-1)
    def policy_mean(x):
        return actor.command_mean(x) if args.command_space_exploration else actor(x)[:,:action_dimensions]
    reports=[];executed=0;canonical_completions=0
    for update in range(1,args.updates+1):
        if time.monotonic()>=deadline or (args.output/'STOP').exists():break
        tick=time.monotonic(); storage=[];infos=[];canonical_stats=[];prior_pairs=[];prior_rewards=[]
        for step in range(args.steps):
            with torch.no_grad():
                x=env.observe().clone()
                full_mean=None if args.command_space_exploration else actor(x)
                mean=actor.command_mean(x) if args.command_space_exploration else full_mean[:,:action_dimensions]
                dist=torch.distributions.Normal(mean,noise)
                action=dist.sample();logprob=dist.log_prob(action).sum(-1);v=value(x)
                if args.command_space_exploration:applied=actor.from_commands(x,action)
                else:
                    applied=full_mean.clone();applied[:,:action_dimensions]=action
                prior_before=env.motion_prior_observation() if motion_prior is not None else None
                _,reward,done,info=env.step(actor.target(applied))
                if motion_prior is not None:
                    pair=torch.cat((prior_before,info['motion_prior_next']),-1)
                    style=motion_prior.reward(pair,info['failed'])
                    reward=reward+args.motion_prior_weight*style
                    prior_pairs.append(pair);prior_rewards.append(style.mean())
                timeout=torch.zeros_like(reward)
                if 'final_observation' in info:
                    timeout=value(info['final_observation'])*info['truncated']
                storage.append((x,action,logprob,v,reward,done.float(),timeout,mean))
                infos.append(torch.stack((info['failed'].float().mean(),info['root_error'].mean(),info['leg_error'].mean())))
                if args.canonical_starts:
                    mask=env.canonical_world
                    canonical_stats.append(torch.stack(((info['complete']&~info['failed']&mask).sum(),
                        (info['terminated']&~info['complete']&mask).sum(),info['age'][mask].max())))
        fields=[torch.stack([r[i] for r in storage]) for i in range(8)]
        x,actions,oldlog,values,rewards,dones,timeouts,oldmeans=fields
        with torch.no_grad():
            advantage=generalized_advantages(rewards,values,dones,timeouts,value(env.observe()))
            returns=advantage+values
            advantage=(advantage-advantage.mean())/(advantage.std()+1e-8)
        x=x.flatten(0,1);actions=actions.flatten(0,1);oldlog=oldlog.flatten();returns=returns.flatten();advantage=advantage.flatten()
        oldmeans=oldmeans.flatten(0,1)
        prior_status=None
        if motion_prior is not None:
            prior_status=motion_prior.update(torch.cat(prior_pairs))
            prior_status['style_reward_mean']=float(torch.stack(prior_rewards).mean())
        # A policy KL stop must not prevent the independent value model from
        # fitting its rollout returns. Warmup trains value only, with fixed actor.
        if critic_optimizer is not None:
            for _ in range(2):
                for batch in torch.randperm(len(x),device=device).split(512):
                    vloss=(value(x[batch])-returns[batch]).square().mean()
                    critic_optimizer.zero_grad(set_to_none=True);vloss.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(),1);critic_optimizer.step()
        kl_value=0.
        rejected=0;accepted_minibatches=0
        for epoch in range(4 if update>args.critic_warmup_updates else 0):
            indices=torch.randperm(len(x),device=device)
            for batch in indices.split(512):
                mean=policy_mean(x[batch]);dist=torch.distributions.Normal(mean,noise[:1])
                logprob=dist.log_prob(actions[batch]).sum(-1)
                ratio=(logprob-oldlog[batch]).exp()
                policy=-torch.minimum(ratio*advantage[batch],ratio.clamp(.8,1.2)*advantage[batch]).mean()
                vloss=policy.new_zeros(()) if critic_optimizer is not None else (value(x[batch])-returns[batch]).square().mean()
                # Recovery examples remain auxiliary; rollout rewards select
                # behavior. This is not a continued supervised-fit recipe.
                imitation=policy.new_zeros(())
                if args.imitation_weight:
                    ei=torch.randint(len(expert_x),(128,),device=device)
                    imitation=((actor.target(actor(expert_x[ei]))[:,:action_dimensions]-expert_y[ei,:action_dimensions])/.25).square().mean()
                loss=policy+.5*vloss+args.imitation_weight*imitation
                old_actor=copy.deepcopy(actor.state_dict());old_critic=copy.deepcopy(critic.state_dict())
                old_optimizer=copy.deepcopy(optimizer.state_dict())
                optimizer.zero_grad(set_to_none=True);loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(),1);nn.utils.clip_grad_norm_(critic.parameters(),1)
                optimizer.step()
                with torch.no_grad():kl_value=float(((policy_mean(x[batch])-oldmeans[batch]).square()/(2*noise[:1].square())).sum(-1).mean())
                if kl_value>.02:
                    actor.load_state_dict(old_actor);critic.load_state_dict(old_critic);optimizer.load_state_dict(old_optimizer)
                    # Spending the rollout's cumulative KL budget is a normal
                    # early stop. Reduce the step size only when the first
                    # proposed step itself is too large, not on every budget
                    # exhaustion. Accepted policies still obey the same0.02 cap.
                    if accepted_minibatches==0:
                        for group in optimizer.param_groups[:-1]:group['lr']=max(1e-7,group['lr']*.5)
                    rejected+=1
                    break
                accepted_minibatches+=1
            if rejected:break
        executed=update
        elapsed=time.monotonic()-tick
        status={'status':'training','update':update,'limit':args.updates,'update_seconds':elapsed,
            'controlled_states_per_second':args.steps*env.count/elapsed,'elapsed_seconds':time.monotonic()-started,
            'reward_mean':float(rewards.mean()),'failure_fraction_root_error_leg_error':torch.stack(infos).mean(0).tolist(),
            'proposed_analytic_kl':kl_value,'rejected_minibatches':rejected,
            'accepted_minibatches':accepted_minibatches,
            'actor_learning_rates':[g['lr'] for g in optimizer.param_groups[:-1]],
            'source_episode_seconds':env.source_duration*.02,'simulation_ready':False}
        with torch.no_grad():
            residual=value(x)-returns
            status['value_mse']=float(residual.square().mean())
            status['value_explained_variance']=float(1-residual.var()/returns.var().clamp_min(1e-8))
            status['actor_frozen_for_value_warmup']=update<=args.critic_warmup_updates
        if prior_status is not None:status['motion_prior']=prior_status
        if canonical_stats:
            stats=torch.stack(canonical_stats)
            canonical_completions+=int(stats[:,0].sum())
            status.update(canonical_training_completions=canonical_completions,
                canonical_training_failures_this_update=int(stats[:,1].sum()),
                longest_canonical_episode_seconds_this_update=float(stats[:,2].max())*.02)
            status['current_longest_episode_seconds_by_clip']={env.meta['clips'][int(ci)]['name']:
                float(env.age[env.canonical_world&(env.clips==ci)].max())*.02 for ci in env.canonical_assignment.unique()}
        write(args.output/'running.json',status)
        if update%5==0:print(json.dumps(status),flush=True)
        if update in (args.checkpoint_updates or [args.updates//2,args.updates]):
            checkpoint=save(update)
            evaluation_status=('native_fixed_physical_trials' if getattr(args,'controller_memory_bank',None)
                               else 'native_full_motion_evaluation')
            write(args.output/'running.json',{**status,'status':evaluation_status})
            report=evaluate_candidate(checkpoint,args.output/f'eval_{update:05d}')
            reports.append({'update':update,'physical_completions':report['physical_completions'],
                'total_pilot_update':args.prior_updates+update,
                'physical_seconds':report['physical_seconds'],'all_tracking_and_hold_passed':report['all_passed']})
            write(args.output/'checkpoint_results.json',reports)
            print(json.dumps({'native_evaluation':reports[-1]}),flush=True)
    final=save(executed,'final')
    write(args.output/'running.json',{'status':'finished','updates':executed,'elapsed_seconds':time.monotonic()-started,
        'final':str(final),'checkpoint_results':reports,'simulation_ready':False,'automatic_extension':False})
    if getattr(env,'preview',None) is not None:env.preview.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--firmware',type=Path,required=True);ap.add_argument('--bank',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--num-envs',type=int,default=128)
    ap.add_argument('--steps',type=int,default=64);ap.add_argument('--updates',type=int,default=100)
    ap.add_argument('--bootstrap-updates',type=int,default=300);ap.add_argument('--wall-minutes',type=float,default=60)
    ap.add_argument('--bootstrap-lr',type=float,default=3e-5)
    ap.add_argument('--command-delay-substeps',type=int,choices=[0,1,2,3],default=0)
    ap.add_argument('--neutral-recovery-gate',action='store_true')
    ap.add_argument('--canonical-starts',action='store_true')
    ap.add_argument('--canonical-worlds',type=int,choices=range(1,7),default=2)
    ap.add_argument('--imitation-weight',type=float,default=.05)
    ap.add_argument('--continue-tracking-errors',action='store_true')
    ap.add_argument('--native-physics',action='store_true',help='Native MuJoCo3.2.3 mjbatch with CUDA policy training')
    ap.add_argument('--policy-device',choices=['cuda:0','cpu'],default='cuda:0')
    ap.add_argument('--resume-actor',type=Path)
    ap.add_argument('--task-commands',action='store_true')
    ap.add_argument('--command-space-exploration',action='store_true',help='Sample coordinated17-dimensional commands before the frozen factory mapping')
    ap.add_argument('--full-body-corrections',action='store_true',help='Learn all23 joint corrections and sample28 latent commands')
    ap.add_argument('--motion-prior',type=Path,help='Training-only actual native23 physical transition demonstrations')
    ap.add_argument('--motion-prior-weight',type=float,default=.5)
    ap.add_argument('--native-targets',action='store_true',help='Original native PD and full target bounds; native physics required')
    ap.add_argument('--task-base-checkpoint',type=Path)
    ap.add_argument('--decoupled-critic',action='store_true')
    ap.add_argument('--critic-warmup-updates',type=int,default=0)
    ap.add_argument('--checkpoint-updates',type=int,nargs='+');ap.add_argument('--prior-updates',type=int,default=0)
    mode=ap.add_mutually_exclusive_group()
    mode.add_argument('--native-base',action='store_true');mode.add_argument('--locomotion-base',action='store_true')
    args=ap.parse_args()
    if not np.isfinite(args.imitation_weight) or args.imitation_weight<0:ap.error('imitation weight must be finite and nonnegative')
    if not np.isfinite(args.motion_prior_weight) or args.motion_prior_weight<=0:ap.error('motion prior weight must be finite and positive')
    if args.native_physics and not args.locomotion_base:ap.error('native physics currently requires --locomotion-base')
    if args.policy_device=='cpu' and not args.native_physics:ap.error('CPU policy training requires native physics')
    if args.task_commands and (not args.locomotion_base or args.task_base_checkpoint is None):ap.error('Task commands require --locomotion-base and --task-base-checkpoint')
    if args.critic_warmup_updates<0 or (args.critic_warmup_updates and not args.decoupled_critic):ap.error('Value warmup requires --decoupled-critic and a nonnegative update count')
    try:run(args)
    except Exception:
        args.output.mkdir(parents=True,exist_ok=True)
        write(args.output/'failure.json',{'status':'failed','traceback':traceback.format_exc(),'hardware_commands':False})
        raise
