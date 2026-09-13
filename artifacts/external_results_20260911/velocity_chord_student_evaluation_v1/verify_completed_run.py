"""Saved-output/provenance verification only; root owns independent physics."""
import hashlib
import json
from pathlib import Path
import numpy as np

BASE=Path(__file__).parent
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def exact(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()


def main():
    launch=read(BASE/'evaluation_process/launch_receipt.json');post=read(BASE/'evaluation_process/postrun_hashes.json')
    assert launch['input_hashes']==post
    for path,digest in post.items():assert sha(path)==digest,path
    exit_record=read(BASE/'evaluation_process/exit.json')
    assert exit_record['exit_code']==1 and exit_record['error'] is None
    report=read(BASE/'nominal/report.json');request=read(BASE/'nominal/request.json')
    assert report['trace_sha256']==sha(BASE/'nominal/trace.npz') and report['request_sha256']==sha(BASE/'nominal/request.json')
    assert request['requested_controls']==report['requested_controls']==1569
    assert report['completed_controls']==261 and report['attempted_controls']==262 and report['physics_steps']==2616
    assert not report['full_segment_completed']
    for name in ('canonical_prefix250_parity.json','actual_query250_input_parity.json','actual_query250_ownexport_output_parity.json'):
        parity=read(BASE/name);assert parity['passed'] and all(v['bitexact'] for v in parity['checks'].values())
    with np.load(BASE/'nominal/trace.npz') as trace,np.load(BASE/'nominal/strict_failure_state.npz') as failure:
        assert trace['qpos'].shape==(263,30) and trace['qvel'].shape==(263,29)
        assert trace['control_integration_before'].shape==(262,291)
        assert trace['raw_proposal'].shape==trace['actual_normalized_action'].shape==(262,23)
        assert trace['physics_qpos'].shape==(2617,30) and trace['physics_qvel'].shape==(2617,29)
        assert int(np.sum(trace['physics_substeps']))==2616 and trace['physics_substeps'][-1]==6
        assert exact(trace['qpos'][-1],failure['qpos_at_failure'])
        assert exact(trace['qvel'][-1],failure['qvel_at_failure'])
        assert exact(trace['final_integration'],failure['integration_at_failure'])
        assert trace['physics_time'][-1]==failure['expected_time_at_failure']
        assert failure['global_control']==261
        speed=float(trace['physics_qvel'][-1,9]);assert speed==20.438077824748298
    hold=read(BASE/'post_lifecycle_hold_5s/report.json')
    assert hold['requested_controls']==250 and hold['attempted_controls']==0 and not hold['full_segment_completed']
    assert not (BASE/'post_lifecycle_hold_5s/trace.npz').exists()
    output=dict(kind='failed_canonical_final70000_saved_output_provenance_verification',verification_passed=True,
        controller_qualified=False,independent_physics_audit_in_this_verification=False,
        requested_controls=1569,full_controls=261,attempted_controls=262,actual_native_steps=2616,
        failure_global_control=261,failure_substep=6,failure_joint='left_knee_joint',
        actual_speed_rad_s=speed,native_speed_limit_rad_s=20.,zero_source_controls=True,conditional_hold_unrun=True,
        original_BFM250_and_query250_and_ownexport_parity_pass=True,all_current_and_postrun_pins_exact=len(post),
        wrapper_exit_code=1,wrapper_error=None,expected_nonzero_incomplete_verdict=True,
        artifacts={path.relative_to(BASE).as_posix():sha(path) for path in (
            BASE/'nominal/trace.npz',BASE/'nominal/report.json',BASE/'nominal/request.json',BASE/'nominal/strict_failure_state.npz',
            BASE/'pilot_outcome.json',BASE/'post_lifecycle_hold_5s/report.json',BASE/'evaluation_binding.json',
            BASE/'evaluation_process/exit.json',BASE/'evaluation_process/launch_receipt.json',BASE/'evaluation_process/postrun_hashes.json',
            BASE/'head_witness/report.json',BASE/'head_witness/witness.npz')},
        verification_source_sha256=sha(__file__),physics_steps_executed=0,inference_calls=0)
    (BASE/'completion_verification.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output))


if __name__=='__main__':main()
