"""Check preserved balance function, trainable backbone and full native range."""
import json
from pathlib import Path
import numpy as np
import torch
from gear_sonic.scripts.evaluate_g1_true23_causal_dynamics import load_case,archive,ROOT,BUNDLE,NEW
from gear_sonic.utils.g1_true23_single_policy import SinglePolicy
from gear_sonic.utils.g1_true23_neutral_balance import NeutralBalance

torch.set_num_threads(1)
_,contract,_,_,_=load_case('walk003')
neutral={k:v[:1] for k,v in archive(BUNDLE/'walk003/native_original.npz').items() if k!='fps'}
weights=ROOT/'artifacts/g1_true23_six_hour_replan_20260910_v1/bfmzero_inference_v1/inference.safetensors'
bank=NEW/'causal_dynamics_v1/bank'
expert=archive(bank/'expert.npz')
features=expert['features'];rows=expert['resets']
actor=SinglePolicy(weights,contract,neutral,features.mean(0),features.std(0,ddof=1))
balance=NeutralBalance(weights,contract,neutral)
indices=np.linspace(200,len(rows)-200,12).astype(int)
anchors=[]
for row in rows[indices]:
    clip=('walk003','walk002','pico')[int(row[0])]
    reference=archive(bank/(clip+'.npz'));frame=int(row[1])
    rotation=reference['root_rotation'][frame]
    anchors.append(np.r_[reference['root'][frame,:2],np.arctan2(rotation[1,0],rotation[0,0])])
values=(rows[indices,2:32],rows[indices,32:61],rows[indices,61:84],rows[indices,84:],anchors)
q,v,prior,history,anchor=[torch.as_tensor(np.asarray(a),dtype=torch.float32) for a in values]
x=torch.as_tensor(features[indices])
with torch.no_grad():
    expected=balance.target(q,v,prior,history,anchor)
    initial=actor.target(actor(x))
error=float((expected-initial).abs().max())
assert error<2e-5,error
loss=actor(x).square().mean();loss.backward()
backbone=[p for p in actor.actor_weights.values()]
nonzero=sum(p.grad is not None and bool(p.grad.abs().sum()>0) for p in backbone)
assert nonzero==len(backbone)
with torch.no_grad():
    actor.range_adapter.bias.fill_(60)
    upper=actor.target(actor(x))
    actor.range_adapter.bias.fill_(-60)
    lower=actor.target(actor(x))
assert torch.max(torch.abs(upper-actor.limits[:,1]))<1e-5
assert torch.max(torch.abs(lower-actor.limits[:,0]))<1e-5
result=dict(states=12,max_balance_target_difference_rad=error,
    trainable_backbone_parameter_tensors=len(backbone),nonzero_backbone_gradient_tensors=nonzero,
    full_native_interval_reachable_all23=True,physical_rollout_pass=False)
(NEW/'causal_dynamics_v1/single_policy_initial_checks.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
