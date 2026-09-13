"""Verify old/new target timing inside an advancing native23 GPU control step."""
from pathlib import Path
import sys
import json
import argparse
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from gear_sonic.envs.mjlab.g1_true23_locomotion_conditioned_dynamics import LocomotionConditionedEnv
from gear_sonic.envs.mjlab.g1_true23_factory_dynamics import FactoryNative23Env
torch.set_num_threads(1)
parser=argparse.ArgumentParser();parser.add_argument('--canonical',action='store_true');parser.add_argument('--canonical-worlds',type=int,default=2);args=parser.parse_args()
base=Path('/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911');fw=base/'onboard_factory_firmware_v1'
env=LocomotionConditionedEnv(ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1',
    base/'causal_dynamics_v1/focused_walk002_task_closure_bank_v1',fw/'decoded_configs/policies/human_loco/fsm_human_loco_config.yaml',
    count=32,command_delay_substeps=2,canonical_starts=args.canonical,canonical_worlds=args.canonical_worlds)
canonical=env.canonical_world.clone()
if args.canonical:
    assert int(canonical.sum())==4*args.canonical_worlds
    assert (env.frames[canonical]==11).all()
    torch.testing.assert_close(env.q[canonical],env.states[env.clips[canonical],10,:30],rtol=0,atol=0)
    assert (env.episode_control_limits[canonical]==env.totals[env.clips[canonical]]+1500).all()
    assert torch.equal(env.reset_teacher_valid,~canonical)
    # Replaced actual-start states have no matched expert command. Retain
    # their dynamics rollouts, but exclude stale parent targets from imitation.
    assert len(env.observe()[env.reset_teacher_valid])==32-4*args.canonical_worlds
old=env.last_applied.clone();new=(old+.01).clamp(env.limits[:,0]+.06,env.limits[:,1]-.06)
native=env.control_torque;records=[]
def measured(target):
    substep=env.control_substep
    expected=torch.where((substep<env.delay_steps)[:,None],old,target)
    wanted=FactoryNative23Env.control_torque(env,expected)
    result=native(target)
    torch.testing.assert_close(result,wanted,rtol=0,atol=0)
    records.append(int((substep<env.delay_steps).sum()))
    return result
env.control_torque=measured
before=env.sim.data.time.clone()
env.step(new)
elapsed=env.sim.data.time-before
expected_delayed=32-int(canonical.sum())
assert len(records)==10 and records[0]==expected_delayed and records[2:]==[0]*8
assert 0<records[1]<expected_delayed,records
# Resets can return individual failed worlds to time zero. Surviving worlds
# must execute all ten physical substeps while the old command is held first.
survived=env.age>0
assert survived.any()
torch.testing.assert_close(elapsed[survived],torch.full_like(elapsed[survived],.02),rtol=1e-5,atol=1e-6)
result=dict(held_old_target_worlds_per_substep=records,torque_schedule_exact=True,
    surviving_worlds=int(survived.sum()),canonical_start_worlds=int(canonical.sum()),
    stale_expert_targets_excluded=bool(torch.equal(env.reset_teacher_valid,~canonical)),
    physical_seconds_per_control=.02,physics_paused_for_command_delay=False,passed=True)
name=f'locomotion_canonical{args.canonical_worlds}_delay_check.json' if args.canonical else 'locomotion_delay_check.json'
(fw/name).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
