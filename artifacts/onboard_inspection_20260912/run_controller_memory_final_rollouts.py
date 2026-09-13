"""One foreground post-pilot replay suite; no training or recurring scheduling."""
import argparse, hashlib, json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
FW=Path('/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/onboard_factory_firmware_v1')
CYCLE=FW/'controller_state_ab_v1'


def idle_training_check():
    conflicts=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
        try:words=(entry/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        names=[Path(word.decode(errors='replace')).name for word in words]
        if any(name in names for name in ('run_controller_memory_experiment.py',
                'train_g1_true23_factory_dynamics.py','render_controller_memory_comparison.py','render_native_clock.py')):
            conflicts.append(int(entry.name))
    if conflicts:raise RuntimeError(f'Training/rendering still active: {conflicts}')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pilot',type=Path,default=CYCLE/'corrected_pilot')
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    decision=json.loads((a.pilot/'decision.json').read_text())
    if decision.get('incomplete'):raise ValueError('Pilot unfinished; report it before selecting final comparisons')
    if decision['continue_seed0']:raise ValueError('Positive pilot requires the planned continuation before final candidate selection')
    idle_training_check();a.output.mkdir(parents=True,exist_ok=False)
    candidate=CYCLE/'comparison_seed0_restored'
    subprocess.run([sys.executable,str(Path(__file__).with_name('package_controller_memory_candidate.py')),
        '--actor',str(a.pilot/'seed_0_replayed/actor_00060.onnx'),'--output',str(candidate)],check=True)
    actors={'retained':CYCLE/'retained_controller/controller.onnx','restored_seed0':candidate/'controller.onnx'}
    deadline=datetime.fromisoformat(json.loads((CYCLE/'cycle.json').read_text())['deadline_utc'].replace('Z','+00:00')).timestamp()
    # Three unchanged nominal timing repetitions, then the fixed seed-0 comparison.
    jobs=[('retained','walk002',repeat) for repeat in range(1,4)]
    jobs += [('restored_seed0','walk002',1)]
    reports=[]
    for label,clip,repeat in jobs:
        idle_training_check()
        if time.time()+90>deadline:raise TimeoutError('Original six-hour cycle deadline leaves insufficient time for a full run')
        out=a.output/f'{label}_{clip}_repeat{repeat}'
        command=[sys.executable,str(ROOT/'artifacts/teleop_resume_20260911/run_causal_native_clock.py'),
            '--actor',str(actors[label]),'--output',str(out),'--clip',clip,
            '--task-commands','--native-targets','--native-preview-guard','--native-standing-capture',
            '--native-preview-delay-substeps','6','--input','recorded','--feedback','ground-truth',
            '--bank',str(FW.parent/'causal_dynamics_v1/focused_walk002_task_closure_bank_v1'),
            '--library',str(FW/'factory_clock_v5/libtrue23clock.so'),
            '--native-preview-library',str(FW/'native_preview_backend_v3/libtrue23preview.so'),
            '--affinity','--realtime-priority']
        print(json.dumps(dict(starting=out.name,utc=datetime.now(timezone.utc).isoformat())),flush=True)
        with (a.output/(out.name+'.log')).open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        report=json.loads((out/'report.json').read_text())
        fields=('physical_complete','physical_steps','controller_deadline_misses','plant_finishes_over2ms_late',
            'maximum_plant_finish_lateness_ms','controller_ms_p50_p95_max','continuous_quiet30','source','passed')
        result=dict(name=out.name,path=str(out),actor_sha256=hashlib.sha256(actors[label].read_bytes()).hexdigest(),
            **{key:report[key] for key in fields})
        reports.append(result)
        (a.output/'summary.json').write_text(json.dumps(dict(runs=reports,training_and_rendering_process_check=True,
            candidate_selected=False,pilot_decision=decision,simulation_ready=False),indent=2)+'\n')
        print(json.dumps(result),flush=True)


if __name__=='__main__':main()
