"""Restore the stronger zero-correction controller after failed command fitting."""
import hashlib,json
from pathlib import Path
import torch

FW=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911/onboard_factory_firmware_v1')
source=FW/'native_target_ppo_v1';out=FW/'native_target_initialization_recovery_v1'
out.mkdir(exist_ok=False)
status=json.loads((source/'running.json').read_text())
assert status['status']=='finished' and status['updates']<100
original=torch.load(source/'actor_factory.pt',map_location='cpu',weights_only=False)
fitted=torch.load(source/'actor_bootstrap.pt',map_location='cpu',weights_only=False)
state=original['actor']
# Keep normalizers computed from the same expert observations. Every network
# parameter returns to its pre-fit value; the zero output layer makes this
# normalization change an exact no-op for the actor's physical function.
for key in ('goal_mean','goal_scale'):state[key]=fitted['actor'][key].clone()
assert torch.count_nonzero(state['goal_head.4.weight'])==0
assert torch.count_nonzero(state['goal_head.4.bias'])==0
request=dict(original['request'],bootstrap_updates=0,initialization='restored pre-fit zero-correction actor with expert input normalization')
checkpoint=out/'actor_initial.pt'
torch.save(dict(actor=state,request=request,update=0),checkpoint)
report=dict(reason='Expert target fitting regressed native physical rollouts; select initialization using physical behavior',
    consumed_updates=status['updates'],remaining_updates=200-status['updates'],total_limit=200,
    remaining_checkpoint_updates=[100-status['updates'],200-status['updates']],
    actor_parameters='all pre-fit values',normalization='fitted input statistics, output-identical with zero correction head',
    critic_and_discriminator='fresh; rejected-policy training state discarded',
    source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (source/'actor_factory.pt',source/'actor_bootstrap.pt')},
    checkpoint=str(checkpoint),hardware_commands=False)
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
