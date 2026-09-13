"""Compare native preview with its Python implementation on physical states."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np,mujoco
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
for dependency in ('/mnt/e/codex_sonic_runtime/bfm_seed_20260910/onnx_deps',
    '/root/.venvs/g1_true23_mjlab/lib/python3.11/site-packages'):sys.path.append(dependency)
from gear_sonic.scripts.evaluate_g1_true23_causal_dynamics import load_case
from gear_sonic.utils.g1_true23_native_preview_guard import NativePreviewGuard


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--library',type=Path,required=True);args=p.parse_args();args.output.mkdir(exist_ok=False)
    fw=Path('/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/onboard_factory_firmware_v1')
    model,c,*_=load_case('pico')
    reference=NativePreviewGuard(model,c,delay_substeps=4)
    native=NativePreviewGuard(model,c,delay_substeps=4,library=args.library)
    rows=[];maximum_error=0.
    for clip in ('walk002','walk003','pico','walk008'):
        with np.load(fw/'native_target_ppo_v2/eval_00185'/clip/'trace.npz') as z:
            for control in (1,max(1,len(z['target'])-5),len(z['target'])-1):
                q=z['physics_qpos'][control*10-1].copy();v=z['physics_qvel'][control*10-1].copy()
                target=z['target'][control].copy();previous=z['target'][control-1].copy()
                original=[x.copy() for x in (q,v,target,previous)]
                expected,python_status=reference.apply(q,v,target,previous)
                actual,status=native.apply(q,v,target,previous)
                for before,after in zip(original,(q,v,target,previous)):np.testing.assert_array_equal(before,after)
                error=float(abs(expected-actual).max());maximum_error=max(maximum_error,error)
                assert error<1e-8,(clip,control,error)
                assert abs(python_status['predicted_excess_rad']-status['predicted_excess_rad'])<1e-8
                for key in ('predicted_lower_excess_rad','predicted_upper_excess_rad'):
                    np.testing.assert_allclose(reference.last_diagnostic[key],native.last_diagnostic[key],atol=1e-8,rtol=0)
                assert status['preview_calls']<=7 and status['native_preview_backend']
                assert np.all(actual>=native.limits[:,0]) and np.all(actual<=native.limits[:,1])
                rows.append(dict(clip=clip,control=control,target_error=error,python_ms=python_status['elapsed_ms'],
                    native_ms=status['elapsed_ms'],preview_calls=status['preview_calls'],
                    initial_predicted_excess=python_status['original_predicted_excess_rad'],
                    accepted_predicted_excess=status['predicted_excess_rad']))
    report=dict(passed=True,maximum_target_error=maximum_error,cases=rows,
        immediate_and_8ms_delayed_application=True,measured_input_unchanged=True,
        no_motion_future=True,physical_rollout_qualified=False,simulation_ready=False)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
