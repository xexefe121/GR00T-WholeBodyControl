"""Linux native integration test: injected rejection at setup and during motion.

Uses real controller and native clock; injected outcomes are test evidence only.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('native_runner',ROOT/'artifacts/teleop_resume_20260911/run_causal_native_clock.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(exist_ok=False)
    fw=runner.NEW/'onboard_factory_firmware_v1'
    real_publish=runner.publish_checked
    results=[]
    for reject_control in (0,4):
        def injected(command,diagnostic,context,policy,records,publish):
            if context['control']==reject_control:
                status=dict(command.status)
                status['native_preview_guard']=dict(status['native_preview_guard'],
                    predicted_limits_satisfied=False,injected_test_rejection=True)
                command=SimpleNamespace(targets=command.targets,status=status)
            return real_publish(command,diagnostic,context,policy,records,publish)
        runner.publish_checked=injected
        out=args.output/f'control_{reject_control}'
        sys.argv=[str(Path(runner.__file__)),'--output',str(out),'--clip','pico','--controls','20',
            '--actor',str(fw/'native_target_ppo_v2/actor_00185.onnx'),
            '--bank',str(runner.NEW/'causal_dynamics_v1/focused_walk002_task_closure_bank_v1'),
            '--library',str(fw/'factory_clock_v4/libtrue23clock.so'),
            '--task-commands','--native-targets','--native-preview-guard','--native-standing-capture',
            '--native-preview-delay-substeps','6','--native-preview-library',str(fw/'native_preview_backend_v2/libtrue23preview.so')]
        runner.main()
        report=json.loads((out/'report.json').read_text())
        calls=np.load(out/'controller_calls.npy')
        with np.load(out/'trace.npz') as trace:
            timing=trace['timing']
            assert not np.any(timing[:,4]==reject_control)
        assert not np.any(calls[:,0]==reject_control)
        assert report['stop_reason']=='preview_rejected'
        assert report['preview_rejection']['control']==reject_control
        assert report['preview_rejection']['application_time_ns'] is None
        assert report['preview_rejection']['physics_steps_finishing_after_stop_request']<=1
        assert report['final_quiet']['status']==report['continuous_quiet30']['status']=='not_reached'
        assert report['physical_steps']==0 if reject_control==0 else 40<=report['physical_steps']<200
        rows=[json.loads(line) for line in (out/'preview_diagnostics.jsonl').read_text().splitlines()]
        assert rows[-1]['control']==reject_control and not rows[-1]['published']
        report['injected_test_only']=True
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        results.append(dict(reject_control=reject_control,physical_steps=report['physical_steps'],
                            rejected_command_published=False,normal_report_written=True))
    (args.output/'verification.json').write_text(json.dumps(dict(passed=True,injected_test_only=True,cases=results),indent=2)+'\n')


if __name__=='__main__':main()
