"""Saved JSON/curve arithmetic only; no model, gradient, native or task fit."""
from pathlib import Path
import hashlib, json
import numpy as np

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
FIT = NEW / 'direct_target_causal_context_study_v2/fit'
paths = [FIT / c / n for c in ('blinded','causal') for n in ('ORT64_metrics.json','training_progress.npy')]
paths += [NEW/'direct_target_causal_context_study_v2/source_snapshot_v1/full_state_objective.py',
          NEW/'direct_target_full_state_secants_v1/source_snapshot_v1/generate_secants.py',
          NEW/'direct_target_full_state_secants_v1/generation/report.json',
          NEW/'direct_target_full_state_data_root_review_v1/results_v1/report.json',
          NEW/'direct_target_context_pair_fit_independent_v1/results_v1/report.json']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
pins = {p.as_posix():sha(p) for p in paths}
groups = ['root_position','root_rotation','joint_position','root_linear_velocity','root_angular_velocity','joint_velocity']
results = {}
for condition in ('blinded','causal'):
    m = json.loads((FIT/condition/'ORT64_metrics.json').read_text())
    progress = np.load(FIT/condition/'training_progress.npy',allow_pickle=False)
    assert progress.shape == (3000,6) and progress.dtype == np.float64 and np.isfinite(progress).all()
    energy = np.array([g['zero_response_MSE'] for g in m['full_state_group_comparison']],np.float64)
    error = np.array([g['response_MSE'] for g in m['full_state_group_comparison']],np.float64)
    assert np.all(energy>0) and len(energy)==6
    previous, latest = progress[2000:2500,:4].mean(0),progress[2500:3000,:4].mean(0)
    results[condition] = dict(
        group_comparison=[dict(group=name,zero_response_MSE=float(e),response_MSE=float(f),response_over_zero=float(f/e),
            zero_energy_share=float(e/energy.sum()),absolute_response_loss_share=float(f/error.sum()),
            illustrative_fixed_energy_equalization=float(energy.mean()/e)) for name,e,f in zip(groups,energy,error)],
        largest_over_smallest_zero_energy=float(energy.max()/energy.min()),
        root_velocity_share_of_response=float((error[3]+error[4])/error.sum()),
        odd_error_fraction=m['full_state_odd_MSE']/m['full_state_objective'],
        previous500_mean=previous.tolist(),last500_mean=latest.tolist(),last500_relative_change=(latest/previous-1).tolist(),
        second_update_nominal_over_first=float(progress[1,0]/progress[0,0]))
assert {p.as_posix():sha(p) for p in paths}==pins
result=dict(saved_analysis_complete=True,results=results,input_sha256=pins,
    no_exact_feature_alias_supported_by_completed_generation=True,
    new_feature_condition_number_or_gradient_measurement=False,
    illustrative_weighting_not_selected=True,task_model_calls=0,gradient_calls=0,optimizer_updates=0,native_steps=0,
    interpretation='Finite-response squared errors mix different teacher response energy scales. This supports a conditioning hypothesis, not convergence/capacity/closed-loop attribution.')
with (HERE/'report.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(report_sha256=sha(HERE/'report.json'),results=results)))
