"""Exact saved132 cost and selection checks; forward kinematics, no dynamics."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
BASE=Path(__file__).resolve().parent.parent;NEW=BASE.parent
sys.path.insert(0,str(BASE/'source_draft_v1'))
from prefix_tracking_cost import PrefixTrackingCost
from ranked_admission import admit
from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle,load_motion_override

ROOT=Path('/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
BUNDLE=ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'
REFERENCE=Path('/mnt/e/codex-artifacts/sonic23_teleop_six_hour_20260910/mjbatch_intent_floor_inputs_v1/walk003/reference.npz')
DIAG=NEW/'filtered_all_candidates132_v1'
def load(path):
    with np.load(path,allow_pickle=False) as value:return {key:value[key].copy() for key in value.files}
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def exact(actual,wanted,name):
    actual,wanted=np.asarray(actual),np.asarray(wanted)
    assert actual.shape==wanted.shape and actual.dtype==wanted.dtype,(name,actual.shape,wanted.shape,actual.dtype,wanted.dtype)
    assert actual.tobytes()==wanted.tobytes(),name

def main():
    assert not (BASE/'saved132_cost_equivalence_test_report.json').exists()
    original=json.loads((DIAG/'results/report.json').read_text());saved=load(DIAG/'results/forecasts.npz');knots=load(DIAG/'results/cost_knots.npz')
    assert sha(DIAG/'results/report.json')=='58d2ea9a9e5fc569667a540ec9667cfcf9a6d393a5847a34bc56da1d1ec0cf0c'
    native,c,original_motion,timeline,manifest=load_native_bundle(BUNDLE,'walk003')
    motion,_=load_motion_override(REFERENCE,BUNDLE,'walk003',native,c,original_motion,timeline,manifest)
    scorer=PrefixTrackingCost(native,c,motion);reports=[]
    for case,row in enumerate(original['rows']):
        steps=int(saved['checked_steps'][case]);count=steps//10+1
        forecast={key:saved[key][case,:steps if key in ('physics_torque','physics_actuator_force') else steps+1]
            for key in ('physics_qpos','physics_qvel','physics_time','physics_torque','physics_actuator_force','warning_counts','warning_lastinfo')}
        result,arrays=scorer.score(row['control'],saved['target'][case],forecast,row['feasible'])
        for key in ('state','features','residual','component_cost','state_cost'):
            exact(arrays[key],knots[key][case,:count],key+' case'+str(case))
        for key in ('input_cost','target_reference'):exact(arrays[key],knots[key][case],key+' case'+str(case))
        exact(arrays['state_goal_frame'],knots['state_goal_frames'][case,:count],'state frame')
        exact(arrays['input_goal_frame'],knots['input_goal_frames'][case],'input frame')
        assert result['five_control_prefix_state_and_input_sum']==row['five_control_prefix_state_and_input_sum']
        assert result['feasible_for_cost_ranking']==row['eligible_for_fixed_horizon_comparison']
        reports.append(result)
    assert scorer.calls==132
    comparisons=[]
    for control in range(250,283):
        start=(control-250)*4;targets=saved['target'][start:start+4];calls=[]
        def forecast(index,target):
            calls.append(index);exact(target,targets[index],'rank target order')
            return dict(feasible=original['rows'][start+index]['feasible'],physics_steps=int(saved['checked_steps'][start+index]),tracking_cost=reports[start+index])
        selected,target,records=admit(targets,forecast)
        assert calls==[0,1,2,3] and len(records)==4
        expected=original['control_comparisons'][control-250]['lowest_cost_candidate']
        assert (None if selected is None else records[selected]['name'])==expected
        if selected is not None:exact(target,targets[selected],'selected target')
        comparisons.append(dict(control=control,selected=selected,all_four_evaluated=True))
    # Even exact duplicates are evaluated; strict equality ties retain original order.
    targets=np.zeros((4,23),np.float64);calls=[]
    def tied(index,target):
        calls.append(index)
        return dict(feasible=True,physics_steps=50,tracking_cost=dict(feasible_for_cost_ranking=True,five_control_prefix_state_and_input_sum=1.))
    selected,_,records=admit(targets,tied)
    assert selected==0 and calls==[0,1,2,3] and [r['duplicate_of'] for r in records]==[None,0,0,0]
    def invalid(index,target):
        return dict(feasible=False,physics_steps=50,tracking_cost=dict(feasible_for_cost_ranking=False,five_control_prefix_state_and_input_sum=-100.))
    assert admit(targets,invalid)[0] is None
    def nonfinite(index,target):
        return dict(feasible=True,physics_steps=50,tracking_cost=dict(feasible_for_cost_ranking=True,five_control_prefix_state_and_input_sum=float('nan')))
    try:admit(targets,nonfinite)
    except ValueError:pass
    else:raise AssertionError('Nonfinite eligible score accepted.')
    report=dict(pass_=True,saved_cases=132,all_available_features_residuals_components_state_input_and_reference_arrays_bitexact=True,
        all132_complete_or_censored_scalar_scores_exact=True,all33_diagnostic_rankings_exact=True,
        all_four_order_including_duplicates_checked=True,exact_tie_first_in_order_checked=True,
        infeasible_finite_score_excluded=True,nonfinite_eligible_score_rejected=True,
        new_physics_steps=0,new_forecast_calls=0,actor_calls=0,optimizer_calls=0,forward_kinematics_only=True,
        comparisons=comparisons,source_sha256={name:sha(BASE/'source_draft_v1'/name) for name in ('prefix_tracking_cost.py','ranked_admission.py')},
        input_sha256={str(path):sha(path) for path in (DIAG/'results/report.json',DIAG/'results/forecasts.npz',DIAG/'results/cost_knots.npz',Path(__file__))})
    (BASE/'saved132_cost_equivalence_test_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({key:value for key,value in report.items() if key not in ('comparisons','source_sha256','input_sha256')}))

if __name__=='__main__':main()
