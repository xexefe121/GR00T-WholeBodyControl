"""Foreground, bounded two-seed controller-memory comparison. No scheduler."""
import argparse,copy,gc,hashlib,json,sys,time
from argparse import Namespace
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,mujoco
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
for p in ('/mnt/e/codex_sonic_runtime/bfm_seed_20260910/onnx_deps','/root/.venvs/g1_true23_mjlab/lib/python3.11/site-packages'):sys.path.append(p)
import torch
from gear_sonic.scripts.train_g1_true23_factory_dynamics import run,write
from evaluate_controller_memory import evaluate


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--arm',choices=['legacy','replayed'])
    ap.add_argument('--seed',type=int,choices=[0,1]);ap.add_argument('--output',type=Path)
    ap.add_argument('--run-directory',type=Path,help='Fresh result directory; original six-hour deadline still applies')
    a=ap.parse_args();fw=Path('/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/onboard_factory_firmware_v1')
    root_cycle=fw/'controller_state_ab_v1'
    cycle=a.run_directory or root_cycle
    cycle.mkdir(parents=True,exist_ok=True)
    deadline=datetime.fromisoformat(json.loads((root_cycle/'cycle.json').read_text())['deadline_utc'].replace('Z','+00:00')).timestamp()
    baseline_cache=cycle/'fixed_baseline.json'
    def callback(actor,path,folder,env):
        state=torch.get_rng_state()
        try:
            digest=hashlib.sha256()
            for key,value in sorted(actor.state_dict().items()):digest.update(key.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            identity=digest.hexdigest()
            if a.smoke:
                folder.mkdir(parents=True,exist_ok=False)
                report=dict(physical_completions=0,physical_seconds=0.,all_passed=False,smoke_only=True,simulation_ready=False)
                write(folder/'report.json',report);return report
            if folder.name=='eval_bootstrap' and baseline_cache.exists():
                cached=json.loads(baseline_cache.read_text())
                if cached['report'].get('evaluation_version')!=2:raise ValueError('Baseline must use corrected standing-hold evaluation')
                if cached['actor_sha256']!=identity:raise ValueError('Frozen baseline actor changed between comparison arms')
                folder.mkdir(parents=True,exist_ok=False);report=cached['report']
                write(folder/'report.json',report);write(folder/'reuse.json',dict(source=str(baseline_cache),actor_sha256=identity));return report
            report=evaluate(actor,path,folder,env)
            if folder.name=='eval_bootstrap':write(baseline_cache,dict(actor_sha256=identity,report=report))
            return report
        finally:torch.set_rng_state(state)
    combinations=[(seed,arm) for seed in ([a.seed] if a.seed is not None else [0,1])
        for arm in ([a.arm] if a.arm else ['legacy','replayed'])]
    for seed,arm in combinations:
        output=a.output or cycle/(('smoke_' if a.smoke else '')+f'seed_{seed}_{arm}')
        if output.exists():raise ValueError(f'Refusing to repeat or overwrite an experiment: {output}')
        remaining=(deadline-time.time())/60-60
        if remaining<=0:raise TimeoutError('Final evaluation hour reserved; no new training arm started')
        args=Namespace(firmware=fw,bank=fw.parent/'causal_dynamics_v1/focused_walk002_task_closure_bank_v1',output=output,
            num_envs=128,steps=64,updates=2 if a.smoke else 60,bootstrap_updates=0,wall_minutes=min(90.,remaining),bootstrap_lr=3e-5,
            command_delay_substeps=6,neutral_recovery_gate=False,canonical_starts=True,canonical_worlds=4,imitation_weight=0.,
            continue_tracking_errors=True,native_physics=True,policy_device='cpu',resume_actor=fw/'native_target_ppo_v2/actor_00185.pt',
            task_commands=True,command_space_exploration=True,full_body_corrections=True,motion_prior=fw/'motion_prior_data_v1/transitions.npz',
            motion_prior_weight=.5,native_targets=True,task_base_checkpoint=fw/'native_mjbatch_lifecycle_ppo_v1/actor_00100.pt',
            decoupled_critic=True,critic_warmup_updates=0 if a.smoke else 20,checkpoint_updates=[2] if a.smoke else [60],prior_updates=0,
            native_base=False,locomotion_base=True,seed=seed,controller_memory_bank=root_cycle/'memory_bank',reset_memory=arm,
            runtime_wrapper=root_cycle/'retained_controller/controller.wrapper.json',
            preview_library=fw/'native_preview_backend_v3/libtrue23preview.so',evaluation_callback=callback,resume_training=False)
        write(cycle/'progress.json',dict(stage='smoke' if a.smoke else 'controlled_training',seed=seed,arm=arm,
            output=str(output),started_utc=datetime.now(timezone.utc).isoformat(),simulation_ready=False))
        try:run(args)
        except Exception:
            import traceback
            write(cycle/'failure.json',dict(seed=seed,arm=arm,traceback=traceback.format_exc(),simulation_ready=False));raise
        gc.collect()
        if not a.smoke and not (output/'eval_00060/report.json').exists():
            write(cycle/'decision.json',dict(continue_seed0=False,incomplete=True,seed=seed,arm=arm,
                reason='bounded arm ended before all40 actor updates and evaluation',simulation_ready=False))
            write(cycle/'progress.json',dict(stage='pilot_incomplete',seed=seed,arm=arm,
                training_active=False,automatic_extension=False,simulation_ready=False))
            return
    if not a.smoke and a.arm is None and a.seed is None:
        comparisons=[]
        for seed in [0,1]:
            reports={arm:json.loads((cycle/f'seed_{seed}_{arm}/eval_00060/report.json').read_text()) for arm in ['legacy','replayed']}
            old,new=reports['legacy'],reports['replayed']
            if any(r.get('evaluation_version')!=2 for r in reports.values()):raise ValueError('Decision requires corrected standing trials')
            passed=(new['movement_passed']>old['movement_passed'] and new['physical_completions']>=old['physical_completions']
                and new['standing_passed']>=old['standing_passed'])
            comparisons.append(dict(seed=seed,passed=passed,legacy={k:old[k] for k in ['movement_passed','physical_completions','standing_passed']},
                replayed={k:new[k] for k in ['movement_passed','physical_completions','standing_passed']}))
        write(cycle/'decision.json',dict(continue_seed0=all(r['passed'] for r in comparisons),comparisons=comparisons,
            automatic_new_recipe=False,simulation_ready=False))
        write(cycle/'progress.json',dict(stage='pilot_complete',training_active=False,
            continue_seed0=all(r['passed'] for r in comparisons),automatic_new_recipe=False,simulation_ready=False))


if __name__=='__main__':main()
