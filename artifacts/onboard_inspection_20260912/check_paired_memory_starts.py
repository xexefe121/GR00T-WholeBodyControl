"""Same per-world sampled physical starts despite different reset ordering."""
import argparse,json,sys
from pathlib import Path
import numpy as np,mujoco
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
for p in ('/mnt/e/codex_sonic_runtime/bfm_seed_20260910/onnx_deps','/root/.venvs/g1_true23_mjlab/lib/python3.11/site-packages'):sys.path.append(p)
import torch
from gear_sonic.envs.mjlab.g1_true23_matched_memory_dynamics import MatchedMemoryNative23Env


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    fw=Path('/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/onboard_factory_firmware_v1');results=[]
    for seed in (0,1):
        envs=[MatchedMemoryNative23Env(ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1',
            fw.parent/'causal_dynamics_v1/focused_walk002_task_closure_bank_v1',fw/'decoded_configs/policies/human_loco/fsm_human_loco_config.yaml',
            count=16,device='cpu',canonical_starts=True,canonical_worlds=4,physics_backend='mjbatch',
            memory_bank=fw/'controller_state_ab_v1/memory_bank',reset_memory=arm,
            preview_library=fw/'native_preview_backend_v3/libtrue23preview.so',experiment_seed=seed) for arm in ('legacy','replayed')]
        try:
            for ordinal in range(4):
                envs[0].reset(np.arange(16))
                # Different completion order cannot change another world's RNG.
                for index in reversed(range(16)):envs[1].reset(np.array([index]))
                for name in ('last_reset_row','frames','clips','episode_control_limits','episode_end_frames'):
                    np.testing.assert_array_equal(np.asarray(getattr(envs[0],name)),np.asarray(getattr(envs[1],name)))
                left,lw=envs[0].sim.export_integration(np.arange(16));right,rw=envs[1].sim.export_integration(np.arange(16))
                np.testing.assert_array_equal(left,right);np.testing.assert_array_equal(lw,rw)
                envs[0].sim.forward()
                refreshed,_=envs[0].sim.export_integration(np.arange(16))
                np.testing.assert_array_equal(refreshed,left)
                np.testing.assert_array_equal(envs[0].delay_rng.integers(0,7,16),envs[1].delay_rng.integers(0,7,16))
                assert np.max(abs(envs[0].observe().numpy()-envs[1].observe().numpy()))>0
            for _ in range(10):
                for env in envs:
                    env.sim.data.ctrl[:]=0.;env.sim.step()
                envs[0].sim.forward()  # extra diagnostic read cannot change next physics
                np.testing.assert_array_equal(envs[0].sim.native['qpos'],envs[1].sim.native['qpos'])
                np.testing.assert_array_equal(envs[0].sim.native['qvel'],envs[1].sim.native['qvel'])
            results.append(dict(seed=seed,physical_starts_exact=True,reset_ordinals_checked=4,controller_memory_differs=True,
                scoring_preserves_integration=True,physical_steps_exact=10))
        finally:
            for env in envs:env.preview.close()
    report=dict(passed=True,seeds=results,changed_reward=False,simulation_ready=False)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
