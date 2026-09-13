"""Verify exact native reset ownership and reconstructed first observations."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np,mujoco
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
for p in ('/mnt/e/codex_sonic_runtime/bfm_seed_20260910/onnx_deps','/root/.venvs/g1_true23_mjlab/lib/python3.11/site-packages'):sys.path.append(p)
import torch
from gear_sonic.envs.mjlab.g1_true23_matched_memory_dynamics import MatchedMemoryNative23Env
from gear_sonic.utils.g1_true23_native_targets import NativeTargetActor


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--count',type=int,default=32);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    fw=Path('/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/onboard_factory_firmware_v1')
    cfg=fw/'decoded_configs/policies/human_loco/fsm_human_loco_config.yaml'
    env=MatchedMemoryNative23Env(ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1',
        fw.parent/'causal_dynamics_v1/focused_walk002_task_closure_bank_v1',cfg,count=a.count,device='cpu',
        canonical_starts=True,canonical_worlds=4,physics_backend='mjbatch',memory_bank=fw/'controller_state_ab_v1/memory_bank',
        reset_memory='replayed',preview_library=fw/'native_preview_backend_v3/libtrue23preview.so')
    env.terminate_tracking_errors=False
    actor=NativeTargetActor(fw/'human_loco_trainable_v1/factory_weights.npz',cfg,env.c,fw/'native_mjbatch_lifecycle_ppo_v1/actor_00100.pt')
    saved=torch.load(fw/'native_target_ppo_v2/actor_00185.pt',map_location='cpu',weights_only=False);actor.load_state_dict(saved['actor']);actor.eval()
    maximum=0.;groups=[]
    for clip in range(3):
        available=np.flatnonzero(env.memory['expert_rows'][:,0]==clip)
        rows=available[np.linspace(0,len(available)-1,a.count).astype(int)];ids=np.arange(a.count)
        env.restore_starts(ids,rows,np.full(a.count,clip))
        observed=env.observe().numpy();expected=env.memory['features'][rows]
        error=np.abs(observed-expected);maximum=max(maximum,float(error.max()))
        groups.append(dict(clip=clip,maximum_error=float(error.max()),worst_feature=int(np.argmax(error)%1582)))
        print(json.dumps(groups[-1]),flush=True)
        np.savez_compressed(a.output/f'features_{clip}.npz',observed=observed,expected=expected,rows=rows)
    state,warning=env.sim.export_integration(np.arange(a.count));env.sim.import_integration(np.arange(a.count),state,warning)
    repeated,repeated_warning=env.sim.export_integration(np.arange(a.count))
    integration_error=float(abs(state-repeated).max());assert integration_error==0
    np.testing.assert_array_equal(warning,repeated_warning)
    if maximum>5e-5:raise AssertionError(f'First-observation mismatch {maximum}')
    env.reset(torch.arange(a.count));started=time.perf_counter()
    with torch.no_grad():
        for _ in range(64):env.step(actor.target(actor(env.observe())))
    elapsed=time.perf_counter()-started
    report=dict(passed=True,maximum_feature_error=maximum,feature_cases=groups,integration_error=integration_error,
        controls=64,worlds=a.count,seconds=elapsed,states_per_second=a.count*64/elapsed,simulation_ready=False)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');env.preview.close();print(json.dumps(report),flush=True)


if __name__=='__main__':main()
