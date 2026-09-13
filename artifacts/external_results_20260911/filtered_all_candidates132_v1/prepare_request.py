"""Freeze all132 fixed counterfactuals before any private forecast executes."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
BASE=Path(__file__).resolve().parent;NEW=BASE.parent
FILTER=NEW/'fast_controller_filtered_v1';EXPERT=NEW/'bfm_entry250_actual_oracle_v1'
ROOT=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def load(path):
    with np.load(path,allow_pickle=False) as value:return {key:value[key].copy() for key in value.files}

def main():
    assert not any((BASE/name).exists() for name in ('request.json','cases.npz','source_snapshot_v1','results','process_status.json'))
    trace=load(FILTER/'nominal/trace.npz');rejected=load(FILTER/'nominal/rejected_proposal.npz')
    contract=read(ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json')
    limits=np.asarray(contract['joint_limits']);rows=[]
    assert sha(FILTER/'nominal/trace.npz')=='2d81a694cd70f2fe4dfddcbd2bc46231928392d235c9eaa7847dbb22aa0847f7'
    assert int(rejected['control'])==282 and int(trace['final_recorded_controls'])==282
    for control in range(250,283):
        integration=trace['control_integration_before'][control] if control<282 else rejected['integration_before']
        primary=trace['proposed_target'][control] if control<282 else rejected['target']
        base=trace['base_target'][control] if control<282 else rejected['base_target']
        qpos,qvel=trace['qpos'][control],trace['qvel'][control]
        targets=[primary,np.clip(base,limits[:,0],limits[:,1]),trace['target'][control-1],np.clip(qpos[7:],limits[:,0],limits[:,1])]
        for candidate,target in enumerate(targets):
            assert target.dtype==np.float64 and np.isfinite(target).all() and np.array_equal(target,np.clip(target,limits[:,0],limits[:,1]))
            rows.append(dict(control=control,candidate=candidate,target=target,integration=integration,qpos=qpos,qvel=qvel,
                warning_counts=trace['physics_warning_counts'][control*10],warning_lastinfo=trace['physics_warning_lastinfo'][control*10]))
    arrays={key:np.asarray([row[key] for row in rows]) for key in rows[0]}
    np.testing.assert_array_equal(arrays['control'],np.repeat(np.arange(250,283),4))
    np.testing.assert_array_equal(arrays['candidate'],np.tile(np.arange(4),33))
    previous=load(FILTER/'nominal/private_forecasts.npz');previous_matches=[]
    for index in range(len(previous['control'])):
        case=(int(previous['control'][index])-250)*4+int(previous['candidate'][index])
        np.testing.assert_array_equal(arrays['target'][case],previous['target'][index]);previous_matches.append([case,index])
    np.savez_compressed(BASE/'cases.npz',**arrays)
    inputs=dict(read(FILTER/'frozen_inputs_v2.json')['input_sha256'])
    def bind(path,digest=None):
        path=Path(path);actual=sha(path)
        if digest is not None:assert actual==digest,(str(path),actual,digest)
        inputs[path.as_posix()]=actual
    for path in (FILTER/'frozen_inputs_v2.json',FILTER/'nominal/trace.npz',FILTER/'nominal/rejected_proposal.npz',
        FILTER/'nominal/private_forecasts.npz',FILTER/'nominal/admission_ledger.json',FILTER/'nominal/report.json',
        EXPERT/'frozen_inputs.json',EXPERT/'source_snapshot_v1/run_bfm250_actual_oracle.py',BASE/'cases.npz',
        NEW/'filtered_forecast_cost_source_review_v1/review.json',
        BASE/'run_diagnostic_durable.ps1',Path(__file__)):bind(path)
    draft=BASE/'source_draft_v1';utils=draft/'gear_sonic/utils';utils.mkdir(parents=True,exist_ok=True)
    (draft/'gear_sonic/__init__.py').write_text('"""Frozen diagnostic package; no shared source imports."""\n',encoding='utf-8')
    (utils/'__init__.py').write_text('"""Frozen original qualified expert cost implementation."""\n',encoding='utf-8')
    expert=read(EXPERT/'frozen_inputs.json')
    for name in ('g1_true23_mjbatch_mpc.py','g1_true23_mjbatch_ilqr_core.py','g1_true23_mjbatch_model.py','g1_true23_relative_foot_cost.py'):
        relative='gear_sonic/utils/'+name;source=EXPERT/'source_snapshot_v1'/relative
        bind(source,expert['source_sha256'][relative]);shutil.copyfile(source,utils/name)
    component=FILTER/'source_snapshot_v1/native_forecast.py'
    bind(component,'81af01274f55c703850cb3ec3801bb41912e3f3786440af5099b769b7bd28824')
    shutil.copyfile(component,draft/'native_forecast.py')
    snapshot=BASE/'source_snapshot_v1';sources={}
    for path in sorted(draft.rglob('*.py')):
        ast.parse(path.read_text(encoding='utf-8'));relative=path.relative_to(draft);dest=snapshot/relative
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest);sources[relative.as_posix()]=sha(dest)
    assert len(sources)==8
    # Bind and check every inherited model/reference input before freezing the request.
    for path,digest in inputs.items():assert sha(path)==digest,path
    request=dict(kind='all132_fixed_saved_state_candidate_forecasts_and_exact_cost_components',case_count=132,
        controls=[250,282],candidate_order=['primary','original_BFM','previous_applied','current_q'],
        forecast_horizon_controls=5,forecast_horizon_seconds=.1,private_forecast_calls=132,repeat_forecast_calls=0,
        actual_plant_steps=0,connected_controller=False,controller_selected=False,optimizer_calls=0,actor_calls=0,
        private_forecast_stop_on_first_strict_failure=True,all132_outcomes_retained=True,
        source_sha256=sources,input_sha256=inputs,existing_saved_forecast_comparisons=previous_matches,
        cost_source='exact qualified query250 Native23Tracker.residual and target_reference; H30 constructor unchanged',
        planner_window='c+10',state_knots=list(range(6)),state_goal_frames='c+10+t for t0..5',
        target_knots=list(range(5)),target_goal_frames='c+11+t for t0..4',
        cost_weights=dict(all_joint_limit_margin=.05,all_joint_limit_weight=2000,relative_foot_weight=0,original_horizon=30),
        cost_scope='Exact original components over initial knot plus available t1..5 state knots and t0..4 input terms. A truncated prefix, not the full H30 MPC objective.',
        incomplete_forecast_costs='Keep available raw components. No finite partial sum ranked against a complete feasible100ms prefix.',
        candidate_rank_diagnostic_only=True,counterfactual_states='Each case starts at its own recorded actual filtered state; no recovered or connected trajectory.',
        no_reference_or_weight_changes=True,no_new_labels_or_fitting=True,hardware_authorized=False)
    (BASE/'request.json').write_text(json.dumps(request,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(request_sha256=sha(BASE/'request.json'),case_count=132,sources=len(sources),inputs=len(inputs),
        cases_sha256=sha(BASE/'cases.npz'),run_source_sha256=sources['run_diagnostic.py'],forecasts_executed=0)))

if __name__=='__main__':main()
