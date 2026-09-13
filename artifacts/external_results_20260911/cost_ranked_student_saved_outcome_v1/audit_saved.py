"""Read-only consistency audit of the one completed cost-ranked attempt."""
import hashlib
import json
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent;BASE=OUT.parent/'fast_controller_cost_ranked_v1'
CONTRACT=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json')
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):
    with np.load(path,allow_pickle=False) as value:return {key:value[key].copy() for key in value.files}

def main():
    assert not (OUT/'report.json').exists()
    trace=load(BASE/'nominal/trace.npz');private=load(BASE/'nominal/private_forecasts.npz');cost=load(BASE/'nominal/candidate_costs.npz')
    rejected=load(BASE/'nominal/rejected_proposal.npz');ledger=read(BASE/'nominal/admission_ledger.json');producer=read(BASE/'nominal/report.json');c=read(CONTRACT)
    controls=len(trace['target']);assert controls==430 and len(ledger)==431
    np.testing.assert_array_equal(trace['global_control'],np.arange(controls))
    assert int(rejected['control'])==int(trace['final_recorded_controls'])==int(rejected['actual_recorded_controls_before'])==controls
    for key,other in [('final_integration','integration_before'),('final_previous_action','actual_previous_action_before')]:np.testing.assert_array_equal(trace[key],rejected[other])
    for key,value in trace.items():
        if key.startswith('final_history_'):np.testing.assert_array_equal(value,rejected['actual_history_before_'+key[len('final_history_'):]])
    np.testing.assert_array_equal(trace['qpos'][-1],rejected['actual_qpos_before']);np.testing.assert_array_equal(trace['qvel'][-1],rejected['actual_qvel_before'])
    np.testing.assert_array_equal(trace['physics_warning_counts'][-1],rejected['warning_before']);np.testing.assert_array_equal(trace['physics_warning_lastinfo'][-1],rejected['warning_lastinfo_before'])
    names=sorted(key[len('final_history_'):] for key in trace if key.startswith('final_history_'))
    np.testing.assert_array_equal(np.concatenate([trace['final_history_'+key].reshape(-1) for key in names]),rejected['actual_history_before'])
    for key in ('target','global_control','control_integration_before','control_history_before','control_previous_action_before','action','selected_prefix_cost'):assert len(trace[key])==controls
    starts=selected_samples=rankings=cost_checks=0
    for row in ledger[250:]:
        control=row['control'];actual_start=control*10;records=row['records'];assert len(records)==4
        candidates=[]
        for index,record in enumerate(records):
            assert record['candidate']==index
            witness=record['witness'];idx=witness['forecast_index'];ss,se=private['state_offsets'][idx:idx+2]
            assert private['control'][idx]==cost['control'][idx]==control and private['candidate'][idx]==cost['candidate'][idx]==index
            assert se-ss==witness['physics_steps']+1 and cost['cost_available'][idx]
            np.testing.assert_array_equal(private['target'][idx],record['target'])
            for key,actual in [('physics_qpos','physics_qpos'),('physics_qvel','physics_qvel'),('physics_time','physics_time'),('warning_counts','physics_warning_counts'),('warning_lastinfo','physics_warning_lastinfo')]:
                np.testing.assert_array_equal(private[key][ss],trace[actual][actual_start])
            starts+=1
            mask=cost['state_knot_available'][idx];count=int(mask.sum());assert count==witness['physics_steps']//10+1
            score=float(np.sum(cost['state_cost'][idx,:count])+np.sum(cost['input_cost'][idx])) if count==6 else None
            assert score==witness['tracking_cost']['five_control_prefix_state_and_input_sum'];cost_checks+=1
            if witness['feasible']:
                assert witness['physics_steps']==50 and score is not None and np.isfinite(score)
                assert record['eligible_prefix_cost']==score;candidates.append((score,index))
            else:assert record['eligible_prefix_cost'] is None
        selected=min(candidates)[1] if candidates else None
        assert row['selected']==selected;rankings+=1
        if selected is not None:
            assert trace['selected_candidate'][control]==selected
            record=records[selected];witness=record['witness'];idx=witness['forecast_index'];ss=private['state_offsets'][idx];ts=private['step_offsets'][idx]
            np.testing.assert_array_equal(trace['target'][control],private['target'][idx])
            assert trace['selected_prefix_cost'][control]==record['eligible_prefix_cost']
            for key,actual,is_step in [('physics_qpos','physics_qpos',False),('physics_qvel','physics_qvel',False),('physics_time','physics_time',False),
                ('warning_counts','physics_warning_counts',False),('warning_lastinfo','physics_warning_lastinfo',False),
                ('physics_torque','physics_torque',True),('physics_actuator_force','physics_actuator_torque',True)]:
                pstart=ts if is_step else ss+1;astart=actual_start if is_step else actual_start+1
                np.testing.assert_array_equal(private[key][pstart:pstart+10],trace[actual][astart:astart+10])
            selected_samples+=10
    assert starts==cost_checks==724 and rankings==181 and selected_samples==1800
    failures=[]
    for row in ledger[-1]['records']:
        witness=row['witness'];idx=witness['forecast_index'];end=private['state_offsets'][idx+1]-1
        velocity=private['physics_qvel'][end,6:];ratio=np.abs(velocity)/c['native_velocity'];joint=int(np.argmax(ratio))
        q=private['physics_qpos'][end];tilt=float(np.arccos(np.clip(1-2*(q[4]**2+q[5]**2),-1,1)))
        failures.append(dict(candidate=row['name'],reasons=witness['first_failure']['reasons'],checked_ms=witness['physics_steps']*2,
            final_root_height_m=float(q[2]),final_tilt_rad=tilt,max_speed_joint=c['joint_names'][joint],
            signed_joint_speed_radps=float(velocity[joint]),speed_cap_radps=c['native_velocity'][joint]))
    selected=trace['selected_candidate'];primary_controls=trace['global_control'][selected==0].tolist()
    inputs=[BASE/'nominal'/name for name in ('trace.npz','private_forecasts.npz','candidate_costs.npz','rejected_proposal.npz','admission_ledger.json','report.json')]+[Path(__file__),CONTRACT]
    report=dict(kind='cost_ranked_completed_attempt_saved_consistency',saved_consistency_pass=True,actual_controls=controls,actual_physics_steps=4300,
        all_candidate_initial_saved_states_exact=starts,all_candidate_saved_scalar_costs_exact=cost_checks,
        all_minimum_feasible_and_tie_order_selections_exact=rankings,selected_actual_forecast_substeps_exact=selected_samples,
        final_rejected_full_history_prior_count_integration_warnings_unchanged=True,actual_control_arrays_aligned=True,
        learned_primary_applied_controls=primary_controls,final_rejected_candidates=failures,
        source_controls=producer['source_metrics']['source_controls'],source_metrics=producer['source_metrics'],
        ranked_cycle_ms_p50_p95_max=producer['ranked_attempted_policy_ms_p50_p95_max'],ranked_deadline_misses=producer['ranked_attempted_policy_20ms_deadline_misses'],
        no_hold_reached=True,full_teleoperation_qualified=False,new_physics_steps=0,actor_calls=0,optimizer_calls=0,
        input_sha256={str(path):sha(path) for path in inputs})
    (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({key:value for key,value in report.items() if key not in ('source_metrics','input_sha256')}))

if __name__=='__main__':main()
