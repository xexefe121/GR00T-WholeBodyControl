"""Corrected 30-second standing test for retained fixed movement evaluations."""
import argparse,copy,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np,mujoco
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
for p in ('/mnt/e/codex_sonic_runtime/bfm_seed_20260910/onnx_deps','/root/.venvs/g1_true23_mjlab/lib/python3.11/site-packages'):sys.path.append(p)
import torch
from gear_sonic.utils.g1_true23_native_targets import NativeTargetActor
from gear_sonic.scripts.evaluate_g1_true23_causal_dynamics import load_case
from evaluate_controller_memory import evaluate


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--actor',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--reuse-movement',type=Path);ap.add_argument('--memory-mode',choices=['legacy','replayed'],default='replayed')
    a=ap.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    fw=Path('/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/onboard_factory_firmware_v1')
    _,contract,*_=load_case('walk002')
    actor=NativeTargetActor(fw/'human_loco_trainable_v1/factory_weights.npz',
        fw/'decoded_configs/policies/human_loco/fsm_human_loco_config.yaml',contract,fw/'native_mjbatch_lifecycle_ppo_v1/actor_00100.pt')
    saved=torch.load(a.actor,map_location='cpu',weights_only=False);actor.load_state_dict(saved['actor'],strict=True)
    env=SimpleNamespace(bank=fw.parent/'causal_dynamics_v1/focused_walk002_task_closure_bank_v1',memory_bank=fw/'controller_state_ab_v1/memory_bank')
    report=evaluate(actor,a.actor,a.output,env,memory_mode=a.memory_mode,only_standing=a.reuse_movement is not None)
    if a.reuse_movement is not None:
        prior=json.loads(a.reuse_movement.read_text())
        if prior['memory_mode']!=a.memory_mode:raise ValueError('cannot reuse movement with a different memory mode')
        report['standing_only_report']=copy.deepcopy(report)
        report['cases']=[r for r in prior['cases'] if not r['standing']]+report['cases']
        for key in ('movement_passed','movement_physical','transition_passed'):report[key]=prior[key]
        report['physical_completions']=sum(r['physical_complete'] for r in report['cases'])
        report['physical_seconds']=sum(sum(c['controls'] for c in r['cases'])*.02 for r in report['cases'])
        report['all_passed']=all(r['passed']==64 for r in report['cases'])
        report['reused_movement_transition_report']=str(a.reuse_movement)
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['movement_passed','standing_passed','physical_completions','evaluation_version']}),flush=True)


if __name__=='__main__':main()
