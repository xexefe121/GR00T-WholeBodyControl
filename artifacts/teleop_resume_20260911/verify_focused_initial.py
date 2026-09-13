"""Release a bounded pilot only when its initial exported behavior is retained."""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
import onnxruntime as ort

ROOT=Path(__file__).resolve().parents[2]
CAUSAL=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911/causal_dynamics_v1')


def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',type=Path,required=True)
    p.add_argument('--reference-actor',type=Path);a=p.parse_args()
    end=time.monotonic()+300
    while not (a.pilot/'actor_initial.normalization.npz').exists():
        if time.monotonic()>end:raise TimeoutError('initial export unavailable')
        if (a.pilot/'failure.json').exists():raise RuntimeError((a.pilot/'failure.json').read_text())
        time.sleep(1)
    reference=CAUSAL/'smooth_action_initial_v1'
    with np.load(reference/'fixed_inputs.npz') as z:
        x=np.concatenate((z['moving'][::16],z['standing']),0)
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
    reference_actor=a.reference_actor or reference/'actor_initial.onnx'
    sessions=[ort.InferenceSession(str(path),sess_options=opts,
        providers=['CPUExecutionProvider']) for path in (reference_actor,a.pilot/'actor_initial.onnx')]
    expected,actual=[session.run(None,{'features':x})[0] for session in sessions]
    with np.load(reference_actor.with_suffix('.normalization.npz')) as z:span=z['span']
    error=float(np.max(np.abs(actual-expected)*span))
    if error>2e-4:raise ValueError('initial target parity failed: '+str(error))
    snapshot=a.pilot/'source_snapshot';snapshot.mkdir(exist_ok=False)
    sources=['gear_sonic/scripts/train_g1_true23_causal_dynamics.py',
        'gear_sonic/envs/mjlab/g1_true23_causal_dynamics.py',
        'gear_sonic/envs/mjlab/g1_true23_focused_dynamics.py',
        'gear_sonic/envs/mjlab/g1_true23_direct_body_dynamics.py',
        'gear_sonic/envs/mjlab/g1_true23_single_policy_dynamics.py',
        'gear_sonic/envs/mjlab/g1_true23_balanced_causal_dynamics.py',
        'gear_sonic/utils/g1_true23_single_policy.py',
        'gear_sonic/utils/g1_true23_direct_body_goal.py']
    for source in sources:shutil.copy2(ROOT/source,snapshot/Path(source).name)
    result=dict(initial_export_matches_tested_candidate=True,states=len(x),max_target_error_rad=error,
                reference_actor=str(reference_actor),
                simulation_qualified=False)
    (a.pilot/'initial_export_check.json').write_text(json.dumps(result,indent=2)+'\n')
    (a.pilot/'CONTINUE').write_text('initial exported behavior retained; bounded updates specified in request.json only\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
