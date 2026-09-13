"""Check native packet/FK encodings and real motion-goal influence."""
import json
import numpy as np
import torch
from gear_sonic.scripts.evaluate_g1_true23_causal_dynamics import load_case,archive,ROOT,BUNDLE,NEW
from gear_sonic.utils.g1_true23_direct_body_goal import causal_body_features,DirectBodySinglePolicy
from gear_sonic.utils.g1_true23_single_policy import SinglePolicy

torch.set_num_threads(1)
bank=NEW/'causal_dynamics_v1/bank';packed=archive(bank/'bfm_reference_inputs_v1.npz')
errors=[]
for i,clip in enumerate(('walk003','walk002','pico','walk008')):
    _,c,motion,_,_=load_case(clip)
    for frame in (11,300,min(650,len(motion['joint_pos'])-2)):
        two={k:v[frame-1:frame+1].copy() for k,v in motion.items() if k!='fps'}
        got=causal_body_features(two,c)[-1]
        errors.append(float(np.max(np.abs(got-packed[f'body_{i}'][frame]))))
        changed={k:np.concatenate((v,v[-1:])) for k,v in two.items()}
        changed['joint_pos'][-1]+=1;changed['body_pos_w'][-1]+=123
        np.testing.assert_array_equal(causal_body_features(changed,c)[1],got)
assert max(errors)<2e-4,max(errors)
expert=archive(bank/'expert.npz');x=expert['features'];rows=expert['resets']
neutral={k:v[:1] for k,v in archive(BUNDLE/'walk003/native_original.npz').items() if k!='fps'}
weights=ROOT/'artifacts/g1_true23_six_hour_replan_20260910_v1/bfmzero_inference_v1/inference.safetensors'
direct=DirectBodySinglePolicy(weights,c,neutral,x.mean(0),x.std(0,ddof=1))
plain=SinglePolicy(weights,c,neutral,x.mean(0),x.std(0,ddof=1));plain.load_state_dict(direct.state_dict())
idx=650;clip,frame=rows[idx,:2].astype(int)
base=torch.tensor(x[idx:idx+1]);body=torch.tensor(packed[f'body_{clip}'][frame:frame+1])
with torch.no_grad():
    zero=torch.cat((base,body,torch.zeros(1,1)),1)
    full=torch.cat((base,body,torch.ones(1,1)),1)
    original=plain.target(plain(base));standing=direct.target(direct(zero));motion=direct.target(direct(full))
    retained=float((standing-original).abs().max());influence=float((motion-standing).abs().max())
assert retained<2e-5,retained
assert influence>.01,influence
result=dict(packet_states=12,max_packet_vs_training_body_feature_error=max(errors),
    future_mutation_has_no_effect=True,neutral_goal_target_error_rad=retained,
    direct_body_goal_target_change_rad=influence,physical_rollout_pass=False)
(NEW/'causal_dynamics_v1/direct_body_input_checks.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
