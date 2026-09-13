"""Bounded paired physical exploration test and timeout/role checks, no learning."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from gear_sonic.envs.mjlab.g1_true23_motion_curriculum import MotionCurriculumEnv
from gear_sonic.scripts.train_g1_true23_causal_dynamics import generalized_advantages
from gear_sonic.utils.g1_true23_direct_body_goal import DirectBodySinglePolicy

ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/causal_dynamics_v1')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--num-envs',type=int,default=128)
    p.add_argument('--actor',type=Path,default=BASE/'motion_curriculum_initial_v1/actor_initial.pt');a=p.parse_args()
    a.output.mkdir(exist_ok=False);torch.set_num_threads(1);torch.manual_seed(20260912)
    # A timeout takes V(final), a physical terminal takes zero. Neither may
    # inherit the next reset episode's reward/value even when it is enormous.
    reward=torch.tensor([[1.],[1000.]]);value=torch.tensor([[2.],[500.]])
    done=torch.tensor([[1.],[1.]]);timeout=torch.tensor([[3.],[0.]])
    adv=generalized_advantages(reward,value,done,timeout,torch.tensor([999.]),gamma=.9,lam=.95)
    torch.testing.assert_close(adv,torch.tensor([[1.7],[500.]]))
    physical=generalized_advantages(reward,value,done,torch.zeros_like(timeout),torch.tensor([999.]),gamma=.9,lam=.95)
    torch.testing.assert_close(physical[0],torch.tensor([-1.]))
    bundle=ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'
    env=MotionCurriculumEnv(bundle,BASE/'focused_walk002_task_closure_bank_v1',count=a.num_envs,
        focus_clip='walk002',task_closure_stand=True,settle_objective=True)
    with np.load(bundle/'walk003/native_original.npz') as z:neutral={k:z[k][:1] for k in z.files if k!='fps'}
    saved=torch.load(a.actor,map_location='cpu',weights_only=False)
    actor=DirectBodySinglePolicy(ROOT/'artifacts/g1_true23_six_hour_replan_20260910_v1/bfmzero_inference_v1/inference.safetensors',
        env.c,neutral,np.zeros(1323),np.ones(1323),smooth_action_tau=.02,normalize_range_adapter=True).cuda()
    actor.load_state_dict(saved['actor']);actor.eval()
    ids=torch.arange(64,device=env.device);allids=torch.arange(env.count,device=env.device)
    pool=env.focus_motion_pool[env.expert_resets[env.focus_motion_pool,1]<=env.source_stops[env.focus_id]-100]
    rows=env.expert_resets[pool[torch.linspace(0,len(pool)-1,64,device=env.device).long()]]
    noise=torch.randn(50,env.count,23,device=env.device)
    outcomes=[]
    with torch.no_grad():
        for sigma in (.03,.06):
            env.reset(allids);env.sim.reset(ids);env.restore_rows(ids,rows)
            env.age[ids]=0;env.episode_end_frames[ids]=env.frames[ids]+100
            env.motion_fraction[ids]=env.reference_blend[env.clips[ids],env.frames[ids]]
            env.sim.forward();alive=torch.ones(64,device=env.device,dtype=torch.bool)
            failures=torch.zeros_like(alive);tracking=torch.zeros_like(alive);source=[]
            for step in range(50):
                action=actor(env.observe())+noise[step]*env.noise_radians(sigma)/actor.span
                _,_,done,info=env.step(actor.target(action))
                failures|=alive&info['failed'][:64];tracking|=alive&info['tracking_failed'][:64]
                alive&=~done[:64];source.append(float(info['source_motion'].float().mean()))
            outcomes.append(dict(moving_std_rad=sigma,starts=64,physical_failures=int(failures.sum()),
                tracking_terminations=int(tracking.sum()),survived_first_second=int(alive.sum()),source_fraction=float(np.mean(source))))
        # Force a real moving state's time limit after one physical control.
        env.reset(allids);env.episode_end_frames[0]=env.frames[0]+1
        _,_,done,info=env.step(actor.target(actor(env.observe())))
        assert bool(info['truncated'][0]) and bool(done[0]) and not bool(info['terminated'][0])
        actual_final=info['final_observation'][0];reset=env.observe()[0]
        assert float((actual_final-reset).abs().max())>1e-3
        # Moving roles remain source states through repeated segment resets.
        fractions=[]
        for _ in range(220):
            _,_,_,info=env.step(actor.target(actor(env.observe())))
            fractions.append(float(info['source_motion'].float().mean()))
        assert min(fractions)>=.75-1e-7,min(fractions)
        transitions=[]
        # Exercise states produced by the real sampler, immediately before
        # each handoff. The former acquisition cutoff reset at this boundary,
        # and the former braking pool never contained a pre-boundary state.
        for name,boundary in (('acquisition',int(env.source_starts[env.focus_id])),
                              ('braking',int(env.source_stops[env.focus_id]))):
            chosen=None
            for attempt in range(500):
                env.reset(allids)
                candidates=((env.roles==1)&(env.clips==env.focus_id)&(env.frames==boundary-1)).nonzero().flatten()
                if len(candidates):chosen=int(candidates[0]);break
            if chosen is None:raise AssertionError(name+' sampler never provided the pre-boundary state')
            assert int(env.episode_end_frames[chosen])>boundary
            observed=[]
            for step in range(4):
                _,_,done,info=env.step(actor.target(actor(env.observe())))
                assert not bool(done[chosen]),(name,step,'transition reset at handoff')
                assert int(env.frames[chosen])==boundary+step and int(env.age[chosen])==step+1
                observed.append(bool(info['source_motion'][chosen]))
            assert observed==([False,True,True,True] if name=='acquisition' else [True,False,False,False])
            transitions.append(dict(kind=name,initial_frame=boundary-1,source_motion=observed,
                physical_controls_without_reset=4,sampler_attempts=attempt+1))
    selected=.06 if outcomes[1]['physical_failures']<=outcomes[0]['physical_failures']+3 else .03
    result=dict(actor=str(a.actor),noise_comparison=outcomes,selected_moving_noise_rad=selected,
        selection='retain .06 unless physical failures increase by more than3 of64 paired starts',
        quota_counts=torch.bincount(env.roles).tolist(),minimum_source_fraction=min(fractions),
        timeout_bootstrap_checked=True,actual_pre_reset_observation_checked=True,
        transitions_cross_boundaries=True,transition_rollouts=transitions,
        cuda_peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,num_envs=env.count,
        trained_updates=0,simulation_qualified=False)
    (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
