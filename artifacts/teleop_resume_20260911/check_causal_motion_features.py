"""Compare new tensor/native features on actual moving expert states."""
import json
from pathlib import Path
import numpy as np
import torch
from gear_sonic.envs.mjlab.g1_true23_causal_dynamics import features_torch
from gear_sonic.utils.g1_true23_received_features import features_numpy

p=Path('/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/causal_dynamics_v1/bank')
with np.load(p/'expert.npz') as z:
    rows=z['resets'][np.linspace(0,len(z['resets'])-1,12).astype(int)].astype(np.float32)
meta=json.loads((p/'bank.json').read_text())
refs=[]
for clip in meta['clips']:
    with np.load(p/clip['file']) as z:refs.append({k:z[k].copy() for k in z.files})
lengths=np.array([len(r['joint']) for r in refs]);n=lengths.max()
padded={k:torch.tensor(np.stack([np.concatenate((r[k],np.repeat(r[k][-1:],n-len(r[k]),axis=0))) for r in refs]),dtype=torch.float32)
    for k in refs[0] if k!='states'}
c=json.loads(Path('/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json').read_text())
actual=features_torch(torch.tensor(rows[:,2:32]),torch.tensor(rows[:,32:61]),padded,
    torch.tensor(rows[:,1],dtype=torch.long),torch.tensor(rows[:,0],dtype=torch.long),
    torch.tensor(c['default_q']),torch.tensor(rows[:,61:84]),torch.tensor(rows[:,84:]),torch.tensor(lengths)).numpy()
expected=np.stack([features_numpy(r[2:32],r[32:61],refs[int(r[0])],int(r[1]),np.array(c['default_q']),r[61:84],r[84:]) for r in rows])
np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-5)
result=dict(states=12,nonzero_history=True,tensor_native_match=True,max_absolute_error=float(np.max(np.abs(actual-expected))))
(p.parent/'motion_feature_check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
