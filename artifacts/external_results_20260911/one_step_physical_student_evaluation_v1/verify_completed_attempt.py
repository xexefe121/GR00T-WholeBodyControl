"""Pure saved-evidence verification; no models, native steps or controller calls."""
from pathlib import Path
import json
import hashlib
import numpy as np

BASE=Path(__file__).parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()

report=read(BASE/'nominal/report.json')
process=BASE/'evaluation_process'
exit_record=read(process/'exit.json')
receipt=read(process/'launch_receipt.json')
post=read(process/'postrun_hashes.json')
assert exit_record['exit_code']==1 and exit_record['error'] is None
for path,digest in receipt['input_hashes'].items():assert post[path]==digest==sha(path),path
trace_path=BASE/'nominal/trace.npz'
assert sha(trace_path)==report['trace_sha256']=='38428ae27055be7c2e771c5a23056c858f004ef38f260a0b45a57662d8d0aae3'
trace=np.load(trace_path)
failure=np.load(BASE/'nominal/strict_failure_state.npz')
contract_path=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json')
contract=read(contract_path)
limits=np.asarray(contract['joint_limits'])
for name in ('canonical_prefix250_parity.json','actual_query250_input_parity.json','actual_query250_ownexport_output_parity.json'):
    assert read(BASE/name)['passed'] is True
assert report['completed_controls']==319 and report['attempted_controls']==320 and report['physics_steps']==3193
assert np.array_equal(trace['global_control'],np.arange(320))
assert np.array_equal(trace['physics_substeps'],np.r_[np.full(319,10),3])
assert trace['physics_qpos'].shape==(3194,30) and trace['control_integration_before'].shape==(320,291)
assert exact(trace['final_integration'],failure['integration_at_failure'])
assert exact(trace['physics_qpos'][-1],failure['qpos_at_failure'])
assert exact(trace['physics_qvel'][-1],failure['qvel_at_failure'])
assert exact(trace['raw_proposal'],trace['base_target']+trace['delta'])
assert exact(trace['target'],np.clip(trace['raw_proposal'],limits[:,0],limits[:,1]))
normalized=((trace['target']-np.asarray(contract['default_q']))*np.asarray(contract['kp'])/(.25*np.asarray(contract['training_effort']))).astype(np.float32)
assert exact(trace['actual_normalized_action'],normalized)
assert exact(trace['control_previous_action_before'][1:],trace['action'][:-1])
assert exact(trace['final_previous_action'],trace['action'][-1])
excess=np.maximum(limits[:,0]-trace['physics_qpos'][-1,7:],trace['physics_qpos'][-1,7:]-limits[:,1])
joint=int(np.argmax(excess));assert float(excess[joint])==report['failure']['range_excess']
clipped=np.flatnonzero(np.any(trace['raw_proposal']!=trace['target'],axis=1))
result=dict(kind='owner_saved_attempt_completion_verification',preservation_passed=True,lifecycle_passed=False,
    no_model_calls=True,no_native_steps=True,source_sha256=sha(__file__),trace_sha256=sha(trace_path),
    report_sha256=sha(BASE/'nominal/report.json'),strict_failure_sha256=sha(BASE/'nominal/strict_failure_state.npz'),
    contract_sha256=sha(contract_path),all_launch_pins_unchanged=len(receipt['input_hashes']),
    launch_receipt_sha256=sha(process/'launch_receipt.json'),exit_sha256=sha(process/'exit.json'),
    canonical_prefix_query_and_witness_passed=True,completed_controls=319,attempted_controls=320,physics_steps=3193,
    full291_failure_exact=True,raw_proposal_and_applied_action_logs_exact=True,raw_action_prior_retained=True,
    first_target_clipping_control=int(clipped[0]) if len(clipped) else None,clipped_control_count=len(clipped),
    failure_joint=contract['joint_names'][joint],failure_joint_index=joint,
    failure_position=float(trace['physics_qpos'][-1,7+joint]),failure_velocity=float(trace['physics_qvel'][-1,6+joint]),
    native_joint_limits=limits[joint].tolist(),failure_target=float(trace['target'][-1,joint]),
    failure_raw_proposal=float(trace['raw_proposal'][-1,joint]),failure_excess=float(excess[joint]),
    first_failure=report['failure'],source_controls=0,hold_executed=False)
with (BASE/'completion_verification.json').open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
print(json.dumps(result))
