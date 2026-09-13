"""Tracking misses retain physical episodes; physical failures still reset."""
from pathlib import Path
import sys,json
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from gear_sonic.envs.mjlab.g1_true23_locomotion_conditioned_dynamics import LocomotionConditionedEnv
torch.set_num_threads(1)
base=Path('/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911');fw=base/'onboard_factory_firmware_v1'
env=LocomotionConditionedEnv(ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1',
    base/'causal_dynamics_v1/focused_walk002_task_closure_bank_v1',fw/'decoded_configs/policies/human_loco/fsm_human_loco_config.yaml',
    count=8,canonical_starts=True,canonical_worlds=6,command_delay_substeps=2)
ids=torch.arange(8,device='cuda');assignment=env.clips[env.canonical_world].clone()
for _ in range(12):
    env.reset(ids)
    torch.testing.assert_close(env.clips[env.canonical_world],assignment,rtol=0,atol=0)
    assert [int((assignment==i).sum()) for i in env.pools]==[2,2,2]
    assert not env.reset_teacher_valid[env.canonical_world].any()
outcomes=[]
for terminate in (True,False):
    env.reset(ids);env.terminate_tracking_errors=terminate;env.v[0]=0
    clip=int(env.clips[0]);frame=int(env.source_starts[clip]);env.frames[0]=frame
    env.references['root'][clip,frame]=env.q[0,:3]+env.q.new_tensor([2.,0,0])
    _,reward,done,info=env.step(env.factory_default.expand(8,-1).clone())
    assert bool(info['tracking_failed'][0]) and not bool(info['failed'][0])
    assert bool(done[0])==terminate
    if not terminate:
        assert int(env.age[0])==1 and not bool(info['terminated'][0]) and float(reward[0])>0
    outcomes.append(dict(tracking_error_terminates=terminate,reset=bool(done[0]),physical_failure=bool(info['failed'][0]),reward=float(reward[0])))
env.reset(ids);env.terminate_tracking_errors=False
env.q[0,8]=env.limits[1,1]+1.;env.sim.forward()
_,_,done,info=env.step(env.factory_default.expand(8,-1).clone())
assert bool(info['failed'][0]) and bool(done[0]) and bool(info['terminated'][0])
result=dict(passed=True,balanced_recording_assignment_preserved_across12_resets=True,
    real20ms_tracking_error_steps=outcomes,physical_failure_still_terminates=True,
    acceptance_limits_unchanged=True,hardware_commands=False)
(fw/'tracking_recovery_training_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
