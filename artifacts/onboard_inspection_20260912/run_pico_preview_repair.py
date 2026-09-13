"""Run pinned older Pico benchmark or strict newer candidate, on simulation only."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
NEW=Path('/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911')
FW=NEW/'onboard_factory_firmware_v1'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',choices=['older','strict-newer'],default='older')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    pins=json.loads(Path(__file__).with_name('pico_preview_repair_pins.json').read_text())
    for relative,expected in pins['firmware_files_sha256'].items():
        actual=hashlib.sha256((FW/relative).read_bytes()).hexdigest()
        if actual!=expected:raise ValueError(f'Pinned file changed: {relative}')
    common=[sys.executable,str(ROOT/'artifacts/teleop_resume_20260911/run_causal_native_clock.py'),
            '--output',str(args.output),'--clip','pico','--hold','30','--affinity','--realtime-priority']
    if args.case=='older':
        common+=['--actor',str(FW/'native_mjbatch_lifecycle_ppo_v1/actor_00100.onnx'),
                 '--library',str(FW/'factory_clock_v3/libtrue23clock.so'),
                 '--locomotion-conditioned','--target-filter-alpha','0.9']
    else:
        common+=['--actor',str(FW/'native_target_ppo_v2/actor_00185.onnx'),
                 '--library',str(FW/'factory_clock_v4/libtrue23clock.so'),
                 '--task-commands','--native-targets','--native-preview-guard','--native-standing-capture',
                 '--native-preview-delay-substeps','6','--preview-failure-policy','strict',
                 '--native-preview-library',str(FW/'native_preview_backend_v2/libtrue23preview.so'),
                 '--bank',str(NEW/'causal_dynamics_v1/focused_walk002_task_closure_bank_v1')]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    log=args.output.with_suffix('.stdout.log')
    with log.open('x') as stream:subprocess.run(common,stdout=stream,stderr=subprocess.STDOUT,check=True)
    report=json.loads((args.output/'report.json').read_text())
    print(json.dumps(dict(case=args.case,report=str(args.output/'report.json'),
        stop_reason=report['stop_reason'],physical_seconds=report['physical_steps']*.002,
        physical_complete=report['physical_complete'],tracking_passed=report['tracking_passed'],
        standing_passed=report['standing_passed'],timing_passed=report['timing_passed']),indent=2),flush=True)
    if (args.output/'worker_error.txt').exists() or (args.output/'runner_error.txt').exists():
        raise RuntimeError(f'Simulator execution error; see {log}')


if __name__=='__main__':main()
