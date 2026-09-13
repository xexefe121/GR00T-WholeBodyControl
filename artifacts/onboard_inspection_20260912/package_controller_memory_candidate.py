"""Pin an unchanged actor with its complete runtime wrapper; originals preserved."""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
import numpy as np,mujoco
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
for p in ('/mnt/e/codex_sonic_runtime/bfm_seed_20260910/onnx_deps','/root/.venvs/g1_true23_mjlab/lib/python3.11/site-packages'):sys.path.append(p)
from gear_sonic.scripts.evaluate_g1_true23_causal_dynamics import load_case
from gear_sonic.utils.g1_true23_native_targets import NativeTargetController
from gear_sonic.utils.g1_true23_controller_state import wrapper_contract,reference_configuration


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--actor',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    fw=Path('/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/onboard_factory_firmware_v1')
    actor=a.output/'controller.onnx';shutil.copy2(a.actor,actor)
    model,c,_,_,timeline=load_case('walk002')
    meta=json.loads((fw.parent/'causal_dynamics_v1/focused_walk002_task_closure_bank_v1/bank.json').read_text())
    options=dict(model=model,tasks=meta['tasks'],standing_qpos=timeline['configured_standing_qpos'],now=-.22,
        native_preview_guard=True,native_standing_capture=True,native_preview_delay_substeps=6,
        native_preview_library=fw/'native_preview_backend_v3/libtrue23preview.so')
    controller=NativeTargetController(actor,fw,c,**options);wrapper=wrapper_contract(controller)
    previous=a.actor.with_suffix('.wrapper.json')
    if previous.exists():
        recorded=json.loads(previous.read_text())
        if any(wrapper.get(key)!=value for key,value in recorded.items()):
            raise ValueError('Requested runtime differs from recorded checkpoint wrapper')
        shutil.copy2(previous,a.output/'original.wrapper.json')
    actor.with_suffix('.wrapper.json').write_text(json.dumps(wrapper,indent=2)+'\n')
    references=reference_configuration(controller)
    actor.with_suffix('.reference.json').write_text(json.dumps(references,indent=2)+'\n')
    # Exercise the exact public load path after pinning all settings.
    NativeTargetController(actor,fw,c,**options)
    incompatible=dict(options,root_velocity_feedback=.1)
    try:NativeTargetController(actor,fw,c,**incompatible)
    except ValueError:pass
    else:raise AssertionError('Incompatible velocity feedback was accepted')
    report=dict(actor_source=str(a.actor),actor_sha256=hashlib.sha256(actor.read_bytes()).hexdigest(),
        actor_unchanged=actor.read_bytes()==a.actor.read_bytes(),wrapper=wrapper,reference_configuration=references,
        incompatible_wrapper_rejected=True,simulation_ready=False,selected_for_teleop=False)
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
