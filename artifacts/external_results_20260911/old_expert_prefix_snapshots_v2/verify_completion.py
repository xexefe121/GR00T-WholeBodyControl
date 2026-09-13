"""Bounded saved-array completion verification; no native or inference imports."""
from pathlib import Path
import hashlib
import json
import numpy as np

BASE=Path(__file__).parent;DEST=BASE/'capture'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def archive(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def main():
    comparisons=[]
    def exact(name,a,b):
        a,b=np.asarray(a),np.asarray(b)
        passed=a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
        comparisons.append(dict(name=name,shape=list(a.shape),dtype=str(a.dtype),byte_exact=passed));assert passed,name
    request=read(BASE/'capture_request.json');launch=read(BASE/'capture_process/launch_receipt.json')
    post=read(BASE/'capture_process/postrun_hashes.json');exit_record=read(BASE/'capture_process/exit.json')
    absence=read(BASE/'capture_process/process_absence.json');report=read(DEST/'report.json')
    assert exit_record['exit_code']==0 and exit_record['error'] is None and absence['matching_process_count']==0
    assert launch['input_hashes']==post
    for path,digest in post.items():assert sha(path)==digest,path
    assert read(DEST/'request.json')==request
    assert report['passed'] and report['request_sha256']==sha(BASE/'capture_request.json')
    assert report['source_sha256']==sha(DEST/'capture_source.py')
    assert sha(DEST/'control_snapshots.npz')==report['snapshots_sha256']
    assert sha(DEST/'replayed_samples.npz')==report['replayed_samples_sha256']
    assert report['completed_controls']==1268 and report['snapshots']==1269
    assert report['attempted_native_steps']==report['completed_native_steps']==report['verified_native_steps']==12680
    assert not report['original_full_lifecycle_reexecuted'] and not report['full_source_or_quiet_qualification']
    assert report['new_inference_calls']==report['new_labels']==report['new_perturbations']==0
    original=archive(Path(request['trace']));snap=archive(DEST/'control_snapshots.npz');samples=archive(DEST/'replayed_samples.npz')
    for actual,expected in [('qpos','physics_qpos'),('qvel','physics_qvel'),('time','physics_time')]:
        exact('all12681 '+actual,samples[actual],original[expected][:12681])
    for actual,expected in [('ctrl','physics_torque'),('force','physics_actuator_torque')]:
        exact('all12680 '+actual,samples[actual],original[expected][:12680])
    for actual,expected in [('warning_counts','physics_warning_counts'),('warning_lastinfo','physics_warning_lastinfo')]:
        assert samples[actual].dtype==np.int32 and original[expected].dtype==np.int64
        exact('native warnings explicitly serialized as original int64 '+actual,samples[actual].astype(np.int64),original[expected][:12681])
    expected=np.empty(12681);expected[0]=0.
    for i in range(12680):expected[i+1]=expected[i]+.002
    exact('independent repeated clock',samples['expected_time'],expected)
    exact('actual and repeated clock',samples['time'],expected)
    exact('1269 boundary indices',snap['control'],np.arange(1269,dtype=np.int64))
    assert snap['control_integration_before'].shape==(1269,291) and snap['integration_spec']==8191
    for key,expected_key in [('qpos','physics_qpos'),('qvel','physics_qvel'),('time','physics_time')]:
        exact('1269 snapshot '+key,snap[key],original[expected_key][:12681:10])
    state=snap['control_integration_before']
    assert np.isfinite(state).all()
    exact('full291 decoded time',state[:,0],snap['time'])
    exact('full291 decoded qpos',state[:,1:31],snap['qpos'])
    exact('full291 decoded qvel',state[:,31:60],snap['qvel'])
    exact('full291 preceding applied ctrl',state[1:,89:112],original['physics_torque'][9:12680:10])
    exact('full291 root/joint application forces zero',state[:,112:],np.zeros((1269,179)))
    for key in ('warning_counts','warning_lastinfo'):
        exact('native boundary '+key,snap[key],np.zeros((1269,8),np.int32))
        exact('original serialized boundary '+key,snap['original_'+key],np.zeros((1269,8),np.int64))
    count=np.r_[np.full(1268,10,np.int64),np.zeros(1,np.int64)]
    for key in ('snapshot_native_steps_attempted','snapshot_native_steps_completed','snapshot_native_steps_verified'):
        exact(key,snap[key],count)
    for key in ('snapshot_control_was_started','snapshot_control_fully_verified'):
        exact(key,snap[key],count>0)
    exact('final full291 equals native boundary1268',snap['final_integration'],state[-1])
    fixture=archive(Path(request['fixture']))
    exact('initial full291 equals canonical fixture',state[0],fixture['state_vector'])
    assert str(snap['original_trace_sha256'].item())==request['trace_sha256']==sha(request['trace'])
    assert read(DEST/'initial_parity.json')['passed']
    for name in ('failure.json','partial_samples.npz','partial_snapshots.npz','failure_state.npz'):
        assert not (DEST/name).exists(),name
    result=dict(kind='saved_old_prefix_snapshot_completion_verification',passed=True,
        native_steps_compared=12680,boundary_snapshots_checked=1269,all_saved_array_comparisons= comparisons,
        source_current_and_postrun_pins_verified=len(post),wrapper_exit_code=0,matching_processes=0,
        old_warning_serialization_preserved='int64 original; int32 native values explicitly serialized for comparison',
        complete_full291_native_capture_retained=True,preexisting_full291_endpoint_claim=False,
        original1569_lifecycle_or_quiet_requalification=False,verification_native_steps=0,verification_inference_calls=0,
        artifacts={str(p.relative_to(BASE)):sha(p) for p in (DEST/'report.json',DEST/'control_snapshots.npz',DEST/'replayed_samples.npz',
            DEST/'initial_parity.json',BASE/'capture_request.json',BASE/'capture_process/launch_receipt.json',BASE/'capture_process/exit.json',
            BASE/'capture_process/postrun_hashes.json',BASE/'capture_process/process_absence.json',BASE/'capture_process/root_launch.json')},
        source_sha256=sha(__file__))
    (BASE/'completion_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,samples=12680,snapshots=1269,pins=len(post),
        snapshots_sha256=sha(DEST/'control_snapshots.npz'),report_sha256=sha(DEST/'report.json'),verification_sha256=sha(BASE/'completion_verification.json'))))


if __name__=='__main__':main()
