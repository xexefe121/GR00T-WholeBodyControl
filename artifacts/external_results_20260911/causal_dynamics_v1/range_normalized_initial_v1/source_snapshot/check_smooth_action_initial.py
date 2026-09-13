"""Inspect/export an opt-in smooth native action initialization; no training."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import onnxruntime as ort
import torch

from gear_sonic.utils.g1_true23_direct_body_goal import DirectBodySinglePolicy
from gear_sonic.utils.g1_true23_single_policy import smooth_native_fraction
from gear_sonic.utils.g1_true23_received_features import MeasuredHistory, features_numpy

ROOT=Path(__file__).resolve().parents[2]
BASE=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911/causal_dynamics_v1')
BUNDLE=ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'


def archive(path):
    with np.load(path) as z:return {k:z[k].copy() for k in z.files}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--tau',type=float,default=.02)
    parser.add_argument('--normalize-range-adapter',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(20260912)
    contract=json.loads((BUNDLE/'contract.json').read_text())
    for key in ('default_q','kp','training_effort','joint_limits'):
        contract[key]=np.asarray(contract[key])
    neutral={k:v[:1] for k,v in archive(BUNDLE/'walk003/native_original.npz').items() if k!='fps'}
    expert=archive(BASE/'bank/expert.npz');resets=expert['resets']
    body=archive(BASE/'bank/bfm_reference_inputs_v1.npz')
    # Fixed 128-state subsequence per training clip, matching the diagnosis.
    ids=np.concatenate([np.flatnonzero((resets[:,0]==i)&(resets[:,1]>400))
        [::max(1,int(sum((resets[:,0]==i)&(resets[:,1]>400))/128))][:128] for i in range(3)])
    moving=[]
    for index in ids:
        clip,frame=map(int,resets[index,:2]);frame=min(frame,len(body[f'body_{clip}'])-1)
        moving.append(np.r_[expert['features'][index],body[f'body_{clip}'][frame],body[f'alpha_{clip}'][frame]])
    moving=np.asarray(moving,np.float32)
    weights=ROOT/'artifacts/g1_true23_six_hour_replan_20260910_v1/bfmzero_inference_v1/inference.safetensors'
    old=DirectBodySinglePolicy(weights,contract,neutral,expert['features'].mean(0),expert['features'].std(0))
    candidate=copy.deepcopy(old);candidate.smooth_action_tau=args.tau
    candidate.normalize_range_adapter=args.normalize_range_adapter
    old_source=BASE/'pilot_direct_body_v1/source_snapshot/g1_true23_single_policy.py'
    spec=importlib.util.spec_from_file_location('native23_preserved_single_policy',old_source)
    original_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(original_module)
    hidden=[]
    hook=old.range_adapter.register_forward_pre_hook(lambda module,inputs:hidden.append(inputs[0].detach()))
    with torch.no_grad():
        inputs=torch.from_numpy(moving)
        old_output=old(inputs);h=hidden.pop()
        baseline=old.default+5*old.action_scale*torch.tanh(old.block('_actor.policy.4',h,activate=False))
        new_output=candidate(inputs)
        np.testing.assert_array_equal(old_output.numpy(),original_module.SinglePolicy.forward(old,inputs).numpy())
    hook.remove()
    raw=((baseline-old.limits[:,0])/old.span).detach().requires_grad_()
    saturated=(raw.detach()<0)|(raw.detach()>1)
    mapped=smooth_native_fraction(raw,args.tau)
    derivative=torch.autograd.grad(mapped.sum(),raw,retain_graph=True)[0]
    inward_loss=(mapped-.5).square().sum()
    inward=torch.autograd.grad(inward_loss,raw)[0]
    valid_inward=torch.where(raw.detach()<0,inward<0,inward>0)
    assert saturated.any() and torch.all(derivative[saturated]>1e-8)
    assert torch.all(valid_inward[saturated])
    assert torch.all((mapped>0)&(mapped<1))
    # Actual measured standing states/history from all three prior native runs.
    reference=archive(BASE/'bank/walk003.npz')
    reference={k:np.repeat(v[11:12],2,axis=0) for k,v in reference.items() if k!='states'}
    for key in ('joint_velocity','root_velocity','root_omega','feet_velocity','task_velocity','task_omega'):
        reference[key][:]=0
    standing=[]
    for velocity in (0.,.03,-.03):
        trace=archive(BASE/'direct_body_initial_standing_v1'/f'stand_{velocity:+.2f}.npz')
        history=MeasuredHistory(contract)
        for control,target in enumerate(trace['target']):
            q=trace['qpos'][control*10];v=trace['qvel'][control*10]
            if control in (0,25,100,300,1499):
                base=features_numpy(q,v,reference,1,contract['default_q'],history.prior,history.vector())
                standing.append(np.r_[base,body['body_0'][11],0.])
            history.commit(q,v,target)
    standing=torch.tensor(np.asarray(standing),dtype=torch.float32)
    with torch.no_grad():
        old_stand=old.target(old(standing));new_stand=candidate.target(candidate(standing))
        old_targets=old.target(old_output);new_targets=candidate.target(new_output)
    report=dict(source='original public BFM weights, zero learned adapters; no resumed trained graph',
        samples=len(ids),smooth_action_tau=args.tau,normalize_range_adapter=args.normalize_range_adapter,
        mapping='algebraic smooth clamp; residual before clamp; ordinary derivatives; no STE',
        default_branch_exact_parity=True,default_parity_source=str(old_source),
        saturated_states=int(saturated.any(-1).sum()),saturated_joint_targets=int(saturated.sum()),
        saturated_by_joint=saturated.sum(0).tolist(),
        saturated_min_d_fraction_d_residual=float(derivative[saturated].min()),
        all_saturated_inward_gradients_pass=True,
        moving_target_change_max_rad=float((new_targets-old_targets).abs().max()),
        moving_target_change_rms_rad=float((new_targets-old_targets).square().mean().sqrt()),
        moving_expert_target_rmse_old_rad=float((old_targets-torch.tensor(expert['targets'][ids])).square().mean().sqrt()),
        moving_expert_target_rmse_new_rad=float((new_targets-torch.tensor(expert['targets'][ids])).square().mean().sqrt()),
        standing_samples=len(standing),standing_target_change_max_rad=float((new_stand-old_stand).abs().max()),
        standing_target_change_rms_rad=float((new_stand-old_stand).square().mean().sqrt()),
        physical_validation_pending=True,simulation_qualified=False,hardware_authorized=False)
    np.savez_compressed(args.output/'fixed_inputs.npz',moving=moving,standing=standing.numpy(),expert_row_ids=ids)
    request=dict(kind='direct_body_smooth_native_initialization',features=1749,single_policy=True,
        direct_body_goal=True,smooth_action_tau=args.tau,retained_neutral_balance=False,
        normalize_range_adapter=args.normalize_range_adapter,
        initial_checkpoint=str(weights),resumed_checkpoint=None,optimizer_steps=0,
        normalization_source=str(BASE/'bank/expert.npz'),hardware_authorized=False,simulation_qualified=False,
        mapping=report['mapping'])
    (args.output/'request.json').write_text(json.dumps(request,indent=2)+'\n')
    torch.save(dict(actor=candidate.state_dict(),request=request,initialization_only=True),args.output/'actor_initial.pt')
    torch.onnx.export(candidate.eval(),torch.zeros(1,1749),str(args.output/'actor_initial.onnx'),
        input_names=['features'],output_names=['normalized_target'],
        dynamic_axes={'features':{0:'batch'},'normalized_target':{0:'batch'}},opset_version=17,dynamo=False)
    session=ort.InferenceSession(str(args.output/'actor_initial.onnx'),providers=['CPUExecutionProvider'])
    sample=np.concatenate((moving[:8],standing.numpy()),axis=0)
    with torch.no_grad():expected=candidate(torch.from_numpy(sample)).numpy()
    actual=session.run(None,{'features':sample})[0]
    error=float(np.max(np.abs((actual-expected)*candidate.span.numpy())))
    assert error<2e-4,error
    report['onnx_target_parity_max_rad']=error
    (args.output/'input_gradient_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    snapshot=args.output/'source_snapshot';snapshot.mkdir()
    for path in (Path(__file__),ROOT/'gear_sonic/utils/g1_true23_single_policy.py',ROOT/'gear_sonic/utils/g1_true23_direct_body_goal.py'):
        shutil.copy2(path,snapshot/path.name)
    # Written last: marker consumed by physical evaluators.
    np.savez(args.output/'actor_initial.normalization.npz',span=candidate.span.numpy(),
        default=candidate.default.numpy(),limits=candidate.limits.numpy())
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
