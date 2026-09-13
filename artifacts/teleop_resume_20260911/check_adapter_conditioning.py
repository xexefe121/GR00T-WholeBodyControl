"""One-step adapter conditioning diagnosis, without simulation or training."""
import copy,json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from gear_sonic.utils.g1_true23_direct_body_goal import DirectBodySinglePolicy

ROOT=Path(__file__).resolve().parents[2]
BASE=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911/causal_dynamics_v1')
OUTPUT=BASE/'adapter_conditioning_v1'


def stats(x):
    x=x.detach()
    return dict(shape=list(x.shape),abs_max=float(x.abs().max()),rms=float(x.square().mean().sqrt()),mean=float(x.mean()))


def main():
    OUTPUT.mkdir(exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(20260912)
    saved=torch.load(BASE/'smooth_action_initial_v1/actor_initial.pt',map_location='cpu',weights_only=False)
    bundle=ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'
    c=json.loads((bundle/'contract.json').read_text())
    for key in ('default_q','kp','training_effort','joint_limits'):c[key]=np.asarray(c[key])
    with np.load(bundle/'walk003/native_original.npz') as z:neutral={k:z[k][:1] for k in z.files if k!='fps'}
    weights=ROOT/'artifacts/g1_true23_six_hour_replan_20260910_v1/bfmzero_inference_v1/inference.safetensors'
    actor=DirectBodySinglePolicy(weights,c,neutral,saved['actor']['mean'].numpy(),saved['actor']['scale'].numpy(),smooth_action_tau=.02)
    actor.load_state_dict(saved['actor']);initial=copy.deepcopy(actor.state_dict())
    bank=BASE/'focused_walk002_bank_v1'
    with np.load(bank/'expert.npz') as z:features=z['features'];targets=z['targets'];resets=z['resets']
    rows=np.flatnonzero((resets[:,0]==1)&(resets[:,1]>=361)&(resets[:,1]<1028))
    rows=rows[np.linspace(0,len(rows)-1,64).astype(int)]
    with np.load(bank/'bfm_reference_inputs_v1.npz') as z:
        frames=resets[rows,1].astype(int)
        inputs=np.concatenate((features[rows],z['body_1'][frames],z['alpha_1'][frames,None]),-1)
    x=torch.tensor(inputs,dtype=torch.float32);teacher=torch.tensor(targets[rows],dtype=torch.float32)
    captured={}
    hooks=[actor.range_adapter.register_forward_pre_hook(lambda m,a:captured.update(range_hidden=a[0].detach().clone())),
        actor.goal_adapter[0].register_forward_pre_hook(lambda m,a:captured.update(goal_input=a[0].detach().clone())),
        actor.goal_adapter[-1].register_forward_pre_hook(lambda m,a:captured.update(goal_hidden=a[0].detach().clone()))]
    with torch.no_grad():before=actor(x).detach()
    for hook in hooks:hook.remove()
    std=.03/actor.span
    generator=torch.Generator().manual_seed(917)
    action=before+std*torch.randn(before.shape,generator=generator)
    advantage=torch.linspace(-1.7,1.7,len(x));advantage=(advantage-advantage.mean())/advantage.std()
    old_lp=torch.distributions.Normal(before,std).log_prob(action).sum(-1)
    def loss():
        mean=actor(x)
        ratio=(torch.distributions.Normal(mean,std).log_prob(action).sum(-1)-old_lp).exp()
        policy=-(ratio*advantage).mean()
        imitation=((actor.target(mean)-teacher)**2).mean()
        return policy+10*imitation
    actor.zero_grad();loss().backward()
    gradients={name:dict(norm=float(p.grad.norm()),abs_max=float(p.grad.abs().max())) for name,p in actor.named_parameters() if p.grad is not None}
    selected=sorted(gradients,key=lambda key:gradients[key]['norm'],reverse=True)
    result=dict(batch='64 fixed walk002 actual expert source states; synthetic fixed normalized advantages plus imitation10',
        features={k:stats(v) for k,v in captured.items()},largest_parameter_gradients={k:gradients[k] for k in selected[:12]},cases=[])
    cases=[('goal',1e-8,False),('range',1e-8,False),('both',1e-8,False),('range',1e-10,False),('range',1e-12,False),('both',1e-8,True),('both',1e-6,True),('both',1e-5,True)]
    for selected_group,lr,normalize in cases:
        actor.load_state_dict(initial);actor.zero_grad(set_to_none=True)
        for name,p in actor.named_parameters():p.requires_grad_(name.startswith('goal_adapter.') if selected_group=='goal' else name.startswith('range_adapter.') if selected_group=='range' else name.startswith(('goal_adapter.','range_adapter.')))
        hook=actor.range_adapter.register_forward_pre_hook(lambda m,a:(F.layer_norm(a[0],(a[0].shape[-1],)),)) if normalize else None
        with torch.no_grad():init_delta=float(((actor(x)-before)*actor.span).abs().max())
        parameters=[p for p in actor.parameters() if p.requires_grad]
        optimizer=torch.optim.Adam(parameters,lr=lr)
        optimizer.zero_grad();loss().backward()
        norm=float(torch.nn.utils.clip_grad_norm_(parameters,.5));optimizer.step()
        with torch.no_grad():
            after=actor(x);change=(after-before)*actor.span
            kl=float((.5*((after-before)/std).square().sum(-1)).mean())
            report=dict(group=selected_group,lr=lr,normalized_range_hidden=normalize,initial_target_change_max_rad=init_delta,
                preclip_gradient_norm=norm,target_change_rms_rad=float(change.square().mean().sqrt()),target_change_max_rad=float(change.abs().max()),kl=kl,passes_existing_kl=kl<=.03)
        if hook:hook.remove()
        result['cases'].append(report);print(json.dumps(report),flush=True)
    (OUTPUT/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2),flush=True)


if __name__=='__main__':main()
