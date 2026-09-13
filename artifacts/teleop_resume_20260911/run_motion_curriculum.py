"""One bounded laptop training run with automatic complete native evaluations."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[2]
BASE=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911/causal_dynamics_v1')
WSLROOT='/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof'
WSLBASE='/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/causal_dynamics_v1'


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');os.replace(temporary,path)


def score(report):
    case=report['cases'][0];source=case['source']
    if not case['physical_complete']:return 100.+1-case['completed_controls']/case['requested_controls']
    if not (case['main_quiet']['passed'] and case['hold_quiet']['passed']):return 50.
    return max(source['root_p95_m']/.2,source['yaw_p95_deg']/15,source['leg_rmse_rad']/.15,
        *[x/.12 for x in source['foot_relative_p95_m']],
        *[x/y for x,y in zip(source['hand_head_relative_p95_m'],(.15,.15,.1))])


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--probe',type=Path,default=BASE/'motion_curriculum_probe_v2/report.json')
    p.add_argument('--updates',type=int,default=2000);p.add_argument('--wall-hours',type=float,default=3.)
    p.add_argument('--resume',type=Path);p.add_argument('--prior-updates',type=int,default=0)
    p.add_argument('--baseline-report',type=Path,default=BASE/'task_closure_v2_full_eval/report.json')
    p.add_argument('--deadline-utc',help='Preserve an existing training budget across implementation repairs.')
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    if a.output.parent!=BASE:raise ValueError('campaign must be a direct child of the configured artifact root')
    probe=json.loads(a.probe.read_text());training=a.output/'training'
    assert probe['timeout_bootstrap_checked'] and probe['minimum_source_fraction']>=.75-1e-7
    assert probe['transitions_cross_boundaries'],'transition handoffs must be physically exercised before training'
    deadline=datetime.fromisoformat(a.deadline_utc) if a.deadline_utc else None
    if deadline is not None:
        if deadline.tzinfo is None:raise ValueError('deadline must include a UTC offset')
        a.wall_hours=min(a.wall_hours,(deadline-datetime.now(timezone.utc)).total_seconds()/3600)
        if a.wall_hours<=0:raise ValueError('original training budget has expired')
    initialization=['--actor-init',WSLBASE+'/motion_curriculum_initial_v1/actor_initial.pt']
    reference_actor=BASE/'task_closure_v2_candidate/actor.onnx'
    if a.resume:
        relative=a.resume.resolve().relative_to(BASE.resolve())
        initialization=['--resume',WSLBASE+'/'+relative.as_posix(),'--prior-updates',str(a.prior_updates)]
        reference_actor=a.resume.with_suffix('.onnx')
    env={**os.environ,'PYTHONPATH':str(ROOT)}
    command=['wsl.exe','-d','Ubuntu-22.04','--cd','/','--','bash',WSLROOT+'/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh',
        'env','PYTHONPATH='+WSLROOT,'OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','/root/.venvs/g1_true23_mjlab/bin/python',
        '-m','gear_sonic.scripts.train_g1_true23_causal_dynamics','--output',WSLBASE+'/'+a.output.name+'/training',
        '--bank',WSLBASE+'/focused_walk002_task_closure_bank_v1',*initialization,
        '--single-policy','--direct-body-goal','--smooth-action-tau','.02','--normalize-range-adapter','--task-closure-stand',
        '--separate-gradient-clipping','--focus-clip','walk002','--motion-curriculum','--settle-objective',
        '--actor-lr','5e-9','--adapter-lr','8e-7','--imitation-weight','10','--bootstrap-updates','0',
        '--num-envs',str(probe['num_envs']),'--steps','64','--minibatches','16','--updates',str(a.updates),
        '--training-extra-hold-controls','1500','--wall-hours',str(a.wall_hours),'--moving-noise-rad',str(probe['selected_moving_noise_rad']),
        '--checkpoint-interval','100','--wait-initial-evaluation']
    snapshot=a.output/'source';snapshot.mkdir()
    sources=['gear_sonic/scripts/train_g1_true23_causal_dynamics.py','gear_sonic/scripts/evaluate_g1_true23_causal_dynamics.py',
        'gear_sonic/envs/mjlab/g1_true23_causal_dynamics.py','gear_sonic/envs/mjlab/g1_true23_motion_curriculum.py',
        'gear_sonic/envs/mjlab/g1_true23_focused_dynamics.py','gear_sonic/envs/mjlab/g1_true23_direct_body_dynamics.py',
        'gear_sonic/utils/g1_true23_direct_body_goal.py','gear_sonic/utils/g1_true23_causal_controller.py','gear_sonic/utils/g1_true23_single_policy.py',
        'artifacts/teleop_resume_20260911/run_motion_curriculum.py','artifacts/teleop_resume_20260911/probe_motion_curriculum.py']
    for name in sources:shutil.copy2(ROOT/name,snapshot/Path(name).name)
    write(a.output/'launch.json',dict(command=command,noise_probe=str(a.probe),schedule_created=False,wall_hours=a.wall_hours,
        native_evaluation_every_updates=100,stop_after_consecutive_regressions=2,hardware_authorized=False,
        original_deadline_utc=a.deadline_utc,transition_handoffs='acquisition crosses into source; braking starts before source ends',
        resume=str(a.resume) if a.resume else None,baseline_report=str(a.baseline_report)))
    baseline=json.loads(a.baseline_report.read_text())
    baseline['cases']=[next(x for x in baseline['cases'] if x['clip']=='walk002')]
    if Path(baseline['cases'][0]['actor']).resolve()!=reference_actor.resolve():raise ValueError('baseline report does not evaluate the initialization actor')
    best=score(baseline);regressions=0;evaluated=[];released=False;last_status=0;stop_reason=None
    write(a.output/'best_motion.json',dict(actor=str(reference_actor),report=str(a.baseline_report),score=best,
        qualified=baseline['cases'][0]['passed'],inherited_baseline=True))
    with (a.output/'training.log').open('w') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        deadline_timer=None
        if deadline is not None:
            def stop_at_deadline():
                training.mkdir(exist_ok=True)
                (training/'STOP').write_text('original_training_deadline\n')
            deadline_timer=threading.Timer(max(0,(deadline-datetime.now(timezone.utc)).total_seconds()),stop_at_deadline)
            deadline_timer.daemon=True;deadline_timer.start()
        write(a.output/'running.json',dict(trainer_pid=process.pid,state='initializing',training=str(training)))
        try:
            while True:
                if not released and (training/'actor_initial.normalization.npz').exists():
                    subprocess.run([sys.executable,str(ROOT/'artifacts/teleop_resume_20260911/verify_focused_initial.py'),
                        '--pilot',str(training),'--reference-actor',str(reference_actor)],cwd=ROOT,env=env,check=True)
                    released=True
                checkpoints=sorted(training.glob('actor_[0-9][0-9][0-9][0-9][0-9].normalization.npz')) if training.exists() else []
                for normalization in checkpoints:
                    stem=normalization.name.split('.')[0]
                    if stem in evaluated:continue
                    actor=training/(stem+'.onnx');output=a.output/('eval_'+stem)
                    with (a.output/(stem+'.evaluation.log')).open('w') as eval_log:
                        subprocess.run([sys.executable,'-m','gear_sonic.scripts.evaluate_g1_true23_causal_dynamics',
                            '--actor',str(actor),'--output',str(output),'--bank',str(BASE/'focused_walk002_task_closure_bank_v1'),
                            '--clips','walk002','--hold','30'],cwd=ROOT,env=env,stdout=eval_log,stderr=subprocess.STDOUT,check=True)
                    report=json.loads((output/'report.json').read_text());value=score(report)
                    regressions=regressions+1 if value>best*1.05 else 0
                    if value<best:best=value;write(a.output/'best_motion.json',dict(actor=str(actor),report=str(output/'report.json'),score=value,qualified=report['passed']))
                    evaluated.append(stem)
                    outcome=dict(checkpoint=stem,score=value,best_score=best,consecutive_regressions=regressions,passed=report['passed'],report=str(output/'report.json'))
                    write(a.output/'latest_evaluation.json',outcome);print(json.dumps(outcome),flush=True)
                    if report['passed'] or regressions>=2:
                        stop_reason='walk002_complete_tracking_and_hold_passed' if report['passed'] else 'two_complete_motion_regressions'
                        (training/'STOP').write_text(stop_reason+'\n')
                if time.monotonic()-last_status>60:
                    last_status=time.monotonic();path=training/'metrics.jsonl';latest=None
                    if path.exists():
                        lines=path.read_text().splitlines()
                        if lines:latest=json.loads(lines[-1])
                    state=dict(trainer_pid=process.pid,state='training' if released else 'initializing',
                        last_update=latest['update'] if latest else None,source_fraction=latest['source_motion_fraction'] if latest else None,
                        stage=latest.get('curriculum_stage') if latest else None,evaluated=evaluated,stop_reason=stop_reason)
                    write(a.output/'running.json',state);print(json.dumps(state),flush=True)
                if process.poll() is not None:
                    # The final export is fully written before process exit;
                    # process it before recording completion.
                    remaining=[x for x in training.glob('actor_[0-9][0-9][0-9][0-9][0-9].normalization.npz') if x.name.split('.')[0] not in evaluated]
                    if remaining:continue
                    if process.returncode:raise RuntimeError('trainer failed; see training.log')
                    break
                time.sleep(1)
        finally:
            if deadline_timer is not None:deadline_timer.cancel()
            if process.poll() is None:
                training.mkdir(exist_ok=True)
                (training/'STOP').write_text('orchestrator stopped; finish current update only\n')
        if stop_reason is None and (training/'STOP').exists():stop_reason=(training/'STOP').read_text().strip()
        write(a.output/'outcome.json',dict(trainer_exit=process.returncode,evaluated=evaluated,best_score=best,
            stop_reason=stop_reason or 'bounded_training_finished',general_live_teleop_qualified=False))
        write(a.output/'running.json',dict(state='finished',evaluated=evaluated,stop_reason=stop_reason))


if __name__=='__main__':main()
