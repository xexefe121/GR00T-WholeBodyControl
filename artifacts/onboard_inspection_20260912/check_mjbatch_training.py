"""Native physics parity, reset isolation and actual training-step throughput."""
from pathlib import Path
import sys,time,json,argparse
import numpy as np,mujoco
ROOT=Path(__file__).resolve().parents[2]
sys.path.append('/root/.venvs/g1_true23_mjlab/lib/python3.11/site-packages');sys.path.insert(0,str(ROOT))
import torch
from gear_sonic.envs.mjlab.g1_true23_locomotion_conditioned_dynamics import LocomotionConditionedEnv
from gear_sonic.utils.g1_true23_locomotion_conditioned import LocomotionConditionedActor
parser=argparse.ArgumentParser();parser.add_argument('--device',choices=['cuda:0','cpu'],default='cuda:0');args=parser.parse_args()
torch.set_num_threads(4 if args.device=='cpu' else 1)
base=Path('/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911');fw=base/'onboard_factory_firmware_v1'
cfg=fw/'decoded_configs/policies/human_loco/fsm_human_loco_config.yaml'
env=LocomotionConditionedEnv(ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1',
    base/'causal_dynamics_v1/focused_walk002_task_closure_bank_v1',cfg,count=128,
    command_delay_substeps=2,canonical_starts=True,canonical_worlds=6,physics_backend='mjbatch',device=args.device)
env.terminate_tracking_errors=False
actor=LocomotionConditionedActor(fw/'human_loco_trainable_v1/factory_weights.npz',cfg,env.c['joint_limits']).to(args.device).eval()
sim=env.sim;model=sim.model;ordinary=mujoco.MjData(model)
mujoco.mj_setState(model,ordinary,sim.batch.bind('state')[0].copy(),mujoco.mjtState.mjSTATE_INTEGRATION)
mujoco.mj_forward(model,ordinary)
original_step=sim.step;errors=[]
def compared_step():
    torque=sim.data.ctrl[0].detach().cpu().numpy().astype(np.float64)
    original_step();ordinary.ctrl[:]=torque;mujoco.mj_step(model,ordinary)
    errors.append(float(max(np.max(np.abs(sim.native['qpos'][0]-ordinary.qpos)),np.max(np.abs(sim.native['qvel'][0]-ordinary.qvel)))))
sim.step=compared_step
with torch.no_grad():env.step(actor.target(actor(env.observe())))
sim.step=original_step
assert len(errors)==10 and max(errors)<1e-12,errors
precise=sim.native['qpos'][0].copy();precise_velocity=sim.native['qvel'][0].copy()
env.reset(torch.tensor([1],device=args.device))
np.testing.assert_array_equal(sim.native['qpos'][0],precise)
np.testing.assert_array_equal(sim.native['qvel'][0],precise_velocity)
np.testing.assert_array_equal(env.q[0].cpu().numpy(),precise.astype(np.float32))
ticks=[]
with torch.no_grad():
    for _ in range(32):
        t=time.perf_counter();env.step(actor.target(actor(env.observe())))
        if args.device!='cpu':torch.cuda.synchronize()
        ticks.append(time.perf_counter()-t)
result=dict(passed=True,mujoco_version=mujoco.__version__,native_integration_max_error=max(errors),
    unrelated_reset_preserves_double_precision_state=True,state_projection=False,
    worlds=128,native_cpu_threads=sim.batch.num_threads,policy_device=args.device,
    median_control_batch_seconds=float(np.median(ticks[2:])),controlled_states_per_second=float(128/np.median(ticks[2:])),
    physics_hz=500,control_hz=50,other_training_may_be_running=True,hardware_commands=False)
(fw/('mjbatch_training_cpu_check.json' if args.device=='cpu' else 'mjbatch_training_check.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
