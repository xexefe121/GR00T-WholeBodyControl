"""Saved-array and receipt verification only; no models or physics."""
from pathlib import Path
import hashlib
import json
import numpy as np

BASE=Path(__file__).parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def exact(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def first_difference(a,b):
    for i,(left,right) in enumerate(zip(a,b)):
        if not exact(np.asarray(left),np.asarray(right)):return i
    return None

report=read(BASE/'nominal/report.json');process=BASE/'evaluation_process'
exit_record=read(process/'exit.json');receipt=read(process/'launch_receipt.json');post=read(process/'postrun_hashes.json')
assert exit_record['exit_code']==1 and exit_record['error'] is None
for path,digest in receipt['input_hashes'].items():assert post[path]==digest==sha(path),path
trace_path=BASE/'nominal/trace.npz'
assert sha(trace_path)==report['trace_sha256']=='454b8c577ef64edc1128cd8e5e9d295c2936f2ba854ddd9aeba0286ff112eaa7'
trace=np.load(trace_path);failure=np.load(BASE/'nominal/strict_failure_state.npz')
original_path=BASE.parent/'one_step_physical_student_evaluation_v1/nominal/trace.npz';original=np.load(original_path)
contract_path=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json')
c=read(contract_path);limits=np.asarray(c['joint_limits'])
parity_names=('canonical_prefix250_parity.json','actual_query250_input_parity.json','actual_query250_ownexport_output_parity.json',
    'original_prefix265_and_first_feedback_parity.json','first_feedback_actor_input265_parity.json',
    'first_feedback_lag_history266_parity.json','first_feedback_actor_lag266_parity.json')
for name in parity_names:assert read(BASE/name)['passed'] is True,name
assert report['completed_controls']==314 and report['attempted_controls']==315 and report['physics_steps']==3146
assert np.array_equal(trace['global_control'],np.arange(315));assert np.array_equal(trace['physics_substeps'],np.r_[np.full(314,10),6])
assert trace['physics_qpos'].shape==(3147,30) and trace['control_integration_before'].shape==(315,291)
assert exact(trace['final_integration'],failure['integration_at_failure'])
assert exact(trace['physics_qpos'][-1],failure['qpos_at_failure']);assert exact(trace['physics_qvel'][-1],failure['qvel_at_failure'])
assert exact(trace['raw_proposal'],trace['base_target']+trace['delta'])
assert exact(trace['target'],np.clip(trace['raw_proposal'],limits[:,0],limits[:,1]))
actual=((trace['target']-np.asarray(c['default_q']))*np.asarray(c['kp'])/(.25*np.asarray(c['training_effort']))).astype(np.float32)
assert exact(trace['actual_normalized_action'],actual)
native_clip=trace['raw_proposal']!=trace['target'];mask=native_clip.copy();mask[:250]=False
assert exact(trace['native_target_clip_mask'],native_clip);assert exact(trace['feedback_clip_mask'],mask)
assert exact(trace['raw_combined_action'],trace['action'])
feedback=trace['action'].copy();feedback[mask]=actual[mask]
assert exact(trace['feedback_action'],feedback)
assert exact(trace['feedback_action'][~mask],trace['action'][~mask])
assert exact(trace['control_previous_action_before'][1:],feedback[:-1]);assert exact(trace['previous_action'],trace['control_previous_action_before'])
assert exact(trace['final_previous_action'],feedback[-1])
for name in ('physics_qpos','physics_qvel','physics_time','physics_expected_time','physics_warning_counts','physics_warning_lastinfo'):
    assert exact(trace[name][:2651],original[name][:2651]),name
for name in ('physics_torque','physics_actuator_torque','range_excess','velocity_ratio','effort_ratio','clock_error'):
    assert exact(trace[name][:2650],original[name][:2650]),name
assert exact(trace['control_integration_before'][265],original['control_integration_before'][265])
assert first_difference(trace['control_previous_action_before'],original['control_previous_action_before'])==265
assert first_difference(trace['history'],original['history'])==266
excess=np.maximum(limits[:,0]-trace['physics_qpos'][-1,7:],trace['physics_qpos'][-1,7:]-limits[:,1]);joint=int(np.argmax(excess))
assert float(excess[joint])==report['failure']['range_excess']
rows=np.flatnonzero(np.any(mask,axis=1))
result=dict(kind='owner_clipped_feedback_saved_attempt_verification',preservation_passed=True,lifecycle_passed=False,
    no_model_calls=True,no_native_steps=True,source_sha256=sha(__file__),trace_sha256=sha(trace_path),original_trace_sha256=sha(original_path),
    report_sha256=sha(BASE/'nominal/report.json'),strict_failure_sha256=sha(BASE/'nominal/strict_failure_state.npz'),
    all_launch_pins_unchanged=len(receipt['input_hashes']),launch_receipt_sha256=sha(process/'launch_receipt.json'),exit_sha256=sha(process/'exit.json'),
    parity_receipts={name:sha(BASE/name) for name in parity_names},all_seven_parities_passed=True,
    original2650_native_steps_exact=True,state265_full291_exact=True,first_prior_difference_control=265,first_actor_lag_difference_control=266,
    raw_combined_log_exact=True,unmasked_float32_components_bitexact=True,selected_feedback_components_exact=True,
    feedback_prior_and_final_state_exact=True,first_feedback_clip_control=int(rows[0]),feedback_clip_control_count=len(rows),
    feedback_clip_component_count=int(mask.sum()),completed_controls=314,attempted_controls=315,physics_steps=3146,
    failure_joint=c['joint_names'][joint],failure_joint_index=joint,failure_position=float(trace['physics_qpos'][-1,7+joint]),
    failure_velocity=float(trace['physics_qvel'][-1,6+joint]),native_joint_limits=limits[joint].tolist(),
    failure_target=float(trace['target'][-1,joint]),failure_raw_proposal=float(trace['raw_proposal'][-1,joint]),
    failure_excess=float(excess[joint]),first_failure=report['failure'],source_controls=0,hold_executed=False)
with (BASE/'completion_verification.json').open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
print(json.dumps(result))
