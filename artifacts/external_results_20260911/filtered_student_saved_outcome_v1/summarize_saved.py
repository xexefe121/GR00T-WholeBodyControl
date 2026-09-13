"""Summarize the single completed filtered attempt using saved arrays only."""
import hashlib
import json
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent
BASE=OUT.parent/'fast_controller_filtered_v1'
CONTRACT=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json')
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def load(path):
    with np.load(path,allow_pickle=False) as value:return {key:value[key].copy() for key in value.files}
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    assert not (OUT/'report.json').exists()
    trace=load(BASE/'nominal/trace.npz');private=load(BASE/'nominal/private_forecasts.npz')
    rejected=load(BASE/'nominal/rejected_proposal.npz');ledger=read(BASE/'nominal/admission_ledger.json');contract=read(CONTRACT)
    producer=read(BASE/'nominal/report.json');status=read(BASE/'canonical_process_status.json')
    assert producer['completed_full_controls']==282 and producer['physics_steps']==2820 and len(ledger)==283
    assert status['process_exit_code']==5 and status['raw_python_exit_code']==0
    assert int(trace['final_recorded_controls'])==int(rejected['actual_recorded_controls_before'])==282
    np.testing.assert_array_equal(trace['global_control'],np.arange(282))
    np.testing.assert_array_equal(trace['final_integration'],rejected['integration_before'])
    np.testing.assert_array_equal(trace['final_previous_action'],rejected['actual_previous_action_before'])
    np.testing.assert_array_equal(trace['qpos'][-1],rejected['actual_qpos_before'])
    np.testing.assert_array_equal(trace['qvel'][-1],rejected['actual_qvel_before'])
    named={key[len('final_history_'):]:value for key,value in trace.items() if key.startswith('final_history_')}
    for key,value in named.items():np.testing.assert_array_equal(value,rejected['actual_history_before_'+key])
    np.testing.assert_array_equal(np.concatenate([named[key].reshape(-1) for key in sorted(named)]),rejected['actual_history_before'])
    np.testing.assert_array_equal(trace['physics_warning_counts'][-1],rejected['warning_before'])
    np.testing.assert_array_equal(trace['physics_warning_lastinfo'][-1],rejected['warning_lastinfo_before'])
    assert all(len(trace[key])==282 for key in ('target','control_integration_before','control_history_before','control_previous_action_before','action'))
    selected_checks=0;all_start_checks=0;summary=[]
    for row in ledger[250:]:
        control=row['control'];selected=row['selected'];actual_start=control*10
        for item in row['records']:
            witness=item['witness']
            if witness is None:continue
            idx=witness['forecast_index'];start,stop=private['state_offsets'][idx:idx+2]
            assert private['control'][idx]==control and private['candidate'][idx]==item['candidate']
            np.testing.assert_array_equal(private['target'][idx],item['target'])
            for saved,actual in [('physics_qpos','physics_qpos'),('physics_qvel','physics_qvel'),('physics_time','physics_time'),
                ('warning_counts','physics_warning_counts'),('warning_lastinfo','physics_warning_lastinfo')]:
                np.testing.assert_array_equal(private[saved][start],trace[actual][actual_start])
            assert stop-start==witness['physics_steps']+1;all_start_checks+=1
        if selected is not None:
            witness=row['records'][selected]['witness'];idx=witness['forecast_index']
            ss=private['state_offsets'][idx];ts=private['step_offsets'][idx]
            for saved,actual,offset in [('physics_qpos','physics_qpos',0),('physics_qvel','physics_qvel',0),
                ('physics_time','physics_time',0),('warning_counts','physics_warning_counts',0),
                ('warning_lastinfo','physics_warning_lastinfo',0),('physics_torque','physics_torque',1),
                ('physics_actuator_force','physics_actuator_torque',1)]:
                ps=ts if offset else ss+1
                real=actual_start if offset else actual_start+1
                np.testing.assert_array_equal(private[saved][ps:ps+10],trace[actual][real:real+10])
            selected_checks+=10
        summary.append(dict(control=control,selected=selected,root_height_m=float(trace['qpos'][control,2]),
            root_vertical_velocity_mps=float(trace['qvel'][control,2]),cycle_ms=row['control_loop_ms']))
    failures=[]
    for item in ledger[-1]['records']:
        witness=item['witness'];idx=witness['forecast_index'];end=private['state_offsets'][idx+1]-1
        velocity=private['physics_qvel'][end,6:];ratios=np.abs(velocity)/contract['native_velocity'];joint=int(np.argmax(ratios))
        failures.append(dict(candidate=item['name'],reasons=witness['first_failure']['reasons'],
            private_steps=witness['physics_steps'],milliseconds=2*witness['physics_steps'],
            final_root_height_m=float(private['physics_qpos'][end,2]),max_speed_joint=contract['joint_names'][joint],
            signed_joint_velocity_radps=float(velocity[joint]),joint_speed_cap_radps=contract['native_velocity'][joint],
            speed_ratio=float(ratios[joint]),metadata_worst_joint_index_semantics='position-excess argmax, not speed argmax'))
    moving=np.asarray([row['control_loop_ms'] for row in ledger[250:]])
    inputs=[BASE/'nominal/trace.npz',BASE/'nominal/private_forecasts.npz',BASE/'nominal/rejected_proposal.npz',
        BASE/'nominal/admission_ledger.json',BASE/'nominal/report.json',BASE/'canonical_process_status.json',CONTRACT,Path(__file__)]
    result=dict(kind='saved_array_only_filtered_outcome',saved_bookkeeping_checks_pass=True,
        actual_controls=282,actual_physics_steps=2820,source_controls=0,hold_run=False,
        all_private_candidate_initial_states_checked=all_start_checks,selected_forecast_actual_samples_bitexact=selected_checks,
        rejected_final_history_prior_count_integration_qpos_qvel_warnings_bitexact=True,
        control_array_lengths_aligned=True,rejected_command_applied=False,
        moving_attempted_cycle_ms_p50_p95_max=np.percentile(moving,[50,95,100]).tolist(),
        moving_attempted_deadline_misses=int(np.sum(moving>20)),moving_attempts=len(moving),
        final_candidate_failures=failures,moving_control_timeline=summary,
        conclusion='All applied commands remained within the saved strict native gates, but finite100ms admission did not maintain balance or ensure a feasible next action. All candidates were rejected at282 while the root was descending. No source motion or hardware qualification achieved.',
        no_live_hardware_freeze_or_safe_stop_claim=True,new_physics_steps=0,actor_calls=0,model_updates=0,
        independent_physics_replay_pending=True,input_sha256={str(path):sha(path) for path in inputs})
    (OUT/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({key:value for key,value in result.items() if key not in ('input_sha256','moving_control_timeline')}))

if __name__=='__main__':main()
