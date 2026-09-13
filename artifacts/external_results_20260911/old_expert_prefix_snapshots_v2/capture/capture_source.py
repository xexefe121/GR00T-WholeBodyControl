"""One selected original walk003 prefix replay and full291 boundary capture.

Derived from the root all-control snapshot replay. Differences are the explicit
old trace schema, 1268-control prefix plus final boundary, no assumed endpoint,
unchanged strict oracle and complete partial/failure recording.
"""
import hashlib
import json
from pathlib import Path
import sys
import traceback
import numpy as np
from capture_checks import exact,compare_sample,snapshot_execution_flags
from strict_native import assess

BASE=Path(__file__).resolve().parent.parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def local(path):
    value=str(path).replace('\\','/')
    return Path('/mnt/'+value[0].lower()+value[2:] if sys.platform!='win32' and len(value)>2 and value[1]==':' else value)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def atomic(path,value):
    temporary=Path(path).with_suffix('.tmp');write(temporary,value);temporary.replace(path)
def archive(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def main():
    request_path=BASE/'capture_request.json';request=read(request_path);request_sha=sha(request_path)
    assert request['root_selected'] is True and request['prefix_controls']==1268 and request['requested_snapshots']==1269
    assert request['requested_native_steps']==12680 and request['inference_authorized'] is False
    output=BASE/'capture';output.mkdir(exist_ok=False)
    write(output/'request.json',request);(output/'capture_source.py').write_bytes(Path(__file__).read_bytes())
    attempts=completed=verified=controls=0;boundaries=[];samples=[];data=native=None;expected=0.;spec=None
    state_vector=None;current_control=-1;current_substep=-1
    def rehash():
        assert sha(request_path)==request_sha,'Capture request changed.'
        for path,digest in request['input_hashes'].items():assert sha(local(path))==digest,path
    def capture_state():
        value=np.empty(291);mujoco.mj_getState(native,data,value,spec);return value
    def sample():
        return dict(qpos=data.qpos.copy(),qvel=data.qvel.copy(),ctrl=data.ctrl.copy(),force=data.qfrc_actuator[6:].copy(),
            warning_counts=data.warning.number.copy(),warning_lastinfo=data.warning.lastinfo.copy(),time=float(data.time),expected_time=expected)
    def boundary(control):
        exact(data.qpos,trace['qpos'][control],'control boundary qpos')
        exact(data.qvel,trace['qvel'][control],'control boundary qvel')
        boundaries.append(dict(control=control,integration=capture_state(),time=float(data.time),qpos=data.qpos.copy(),qvel=data.qvel.copy(),
            warning_counts=data.warning.number.copy(),warning_lastinfo=data.warning.lastinfo.copy()))
    def save_samples(path):
        shapes=dict(qpos=(30,),qvel=(29,),ctrl=(23,),force=(23,),warning_counts=(8,),warning_lastinfo=(8,))
        values={}
        for key in ('qpos','qvel','ctrl','force','warning_counts','warning_lastinfo','time','expected_time'):
            # Initial ctrl/force have no preceding command. Retain only actual
            # returned native command samples for their canonical arrays.
            rows=samples[1:] if key in ('ctrl','force') else samples
            value=np.asarray([row[key] for row in rows],dtype=np.int32 if key.startswith('warning') else np.float64)
            if not len(value) and key in shapes:value=value.reshape((0,)+shapes[key])
            values[key]=value
        np.savez_compressed(path,**values)
    def save_boundaries(path,final=None):
        values=dict(control=np.asarray([b['control'] for b in boundaries],np.int64),
            control_integration_before=np.asarray([b['integration'] for b in boundaries],np.float64).reshape(-1,291),
            integration_spec=np.asarray(spec if spec is not None else 8191,np.int64),
            time=np.asarray([b['time'] for b in boundaries],np.float64),
            qpos=np.asarray([b['qpos'] for b in boundaries],np.float64).reshape(-1,30),
            qvel=np.asarray([b['qvel'] for b in boundaries],np.float64).reshape(-1,29),
            warning_counts=np.asarray([b['warning_counts'] for b in boundaries],np.int32).reshape(-1,8),
            warning_lastinfo=np.asarray([b['warning_lastinfo'] for b in boundaries],np.int32).reshape(-1,8),
            original_trace_sha256=np.asarray(request['trace_sha256']),
            **snapshot_execution_flags([b['control'] for b in boundaries],completed,verified,attempts))
        values['original_warning_counts']=values['warning_counts'].astype(np.int64)
        values['original_warning_lastinfo']=values['warning_lastinfo'].astype(np.int64)
        if final is not None:values['final_integration']=final
        np.savez_compressed(path,**values)
    try:
        rehash()
        assert sys.platform!='win32','Use selected pinned WSL native runtime.'
        import mujoco
        from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle
        assert mujoco.__version__=='3.2.3' and np.__version__=='1.26.4'
        trace=archive(local(request['trace']));old_report=read(local(request['original_report']))
        assert sha(local(request['trace']))==request['trace_sha256']==old_report['trace_sha256']
        assert old_report['completed_controls']==1569 and old_report['physics_steps']==15690
        assert old_report['strict_physical_limits_pass'] is True and old_report['failure'] is None
        assert len(trace['target'])==1569 and trace['physics_qpos'].shape==(15691,30)
        assert trace['physics_qvel'].shape==(15691,29) and trace['physics_torque'].shape==(15690,23)
        assert trace['physics_actuator_torque'].shape==(15690,23)
        assert trace['physics_warning_counts'].dtype==trace['physics_warning_lastinfo'].dtype==np.int64
        assert trace['physics_warning_counts'].shape==trace['physics_warning_lastinfo'].shape==(15691,8)
        assert not np.any(trace['physics_warning_counts']) and not np.any(trace['physics_warning_lastinfo'])
        assert np.array_equal(trace['physics_substeps'],np.full(1569,10))
        native,contract,*_=load_native_bundle(local(request['bundle']),'walk003')
        fixture=archive(local(request['fixture']));fixture_report=read(local(request['fixture_report']))
        assert fixture_report['fixture_sha256']==sha(local(request['fixture']))
        assert fixture_report['source_frame']==10 and fixture_report['canonical_control']==0
        initial=fixture['state_vector'].copy();spec=int(fixture['state_spec'])
        assert spec==int(mujoco.mjtState.mjSTATE_INTEGRATION)==8191
        assert mujoco.mj_stateSize(native,spec)==291 and initial.shape==(291,)
        data=mujoco.MjData(native)
        mujoco.mj_setState(native,data,initial,spec);mujoco.mj_forward(native,data);mujoco.mj_setState(native,data,initial,spec)
        expected=float(data.time);state_vector=capture_state()
        samples.append(sample());exact(state_vector,initial,'complete canonical fixture before step0')
        compare_sample(samples[-1],trace,0,expected)
        reasons,metrics=assess(data,contract,expected);assert not reasons,(reasons,metrics)
        write(output/'initial_parity.json',dict(passed=True,full291_fixture_byteexact=True,
            recorded_initial_qpos_qvel_time_and_warning_values_exact=True,physics_steps=0))
        kp,kd,effort=[np.asarray(contract[key]) for key in ('kp','kd','native_effort')]
        for current_control in range(1268):
            assert not np.any(data.qfrc_applied) and not np.any(data.xfrc_applied)
            boundary(current_control)
            target=trace['target'][current_control]
            for current_substep in range(10):
                data.ctrl[:]=np.clip(kp*(target-data.qpos[7:])-kd*data.qvel[6:],-effort,effort)
                attempts+=1
                mujoco.mj_step(native,data)
                completed+=1;expected+=.002;samples.append(sample())
                compare_sample(samples[-1],trace,completed,expected)
                reasons,metrics=assess(data,contract,expected)
                if reasons:raise ValueError('strict native failure: '+repr((current_control,current_substep+1,reasons,metrics)))
                verified+=1
            controls=current_control+1
            if controls%200==0:
                atomic(output/'progress.json',dict(completed_controls=controls,completed_native_steps=completed,
                    verified_native_steps=verified,requested_controls=1268,requested_native_steps=12680))
        boundary(1268);final=capture_state()
        assert controls==1268 and completed==verified==attempts==12680 and len(boundaries)==1269
        exact(final,boundaries[-1]['integration'],'recorded final native boundary')
        save_samples(output/'replayed_samples.npz');save_boundaries(output/'control_snapshots.npz',final)
        rehash()
        report=dict(kind='independent_old_walk003_recorded_prefix_boundary_reconstruction',passed=True,
            requested_controls=1268,completed_controls=controls,requested_native_steps=12680,
            attempted_native_steps=attempts,completed_native_steps=completed,verified_native_steps=verified,
            captured_boundary_controls=[0,1268],snapshots=1269,last_snapshot_control_executed=False,
            integration_spec=8191,integration_size=291,original_full_lifecycle_controls=1569,
            original_full_lifecycle_reexecuted=False,full_source_or_quiet_qualification=False,
            all_recorded_prefix_samples_byteexact=True,warning_comparison='native int32 explicitly serialized to original int64, original file unchanged',
            independent_repeated_clock_exact=True,strict_native_oracle_pass=True,
            final291_reconstructed_from_native_replay=True,preexisting_full291_endpoint_claim=False,
            state_rewrites_after_initialization=0,new_inference_calls=0,new_labels=0,new_perturbations=0,
            original_trace_sha256=request['trace_sha256'],snapshots_sha256=sha(output/'control_snapshots.npz'),
            replayed_samples_sha256=sha(output/'replayed_samples.npz'),request_sha256=request_sha,
            source_sha256=sha(__file__),final_time=float(data.time),input_hashes=request['input_hashes'])
        write(output/'report.json',report);print(json.dumps({k:v for k,v in report.items() if k!='input_hashes'}),flush=True)
    except BaseException as error:
        save_samples(output/'partial_samples.npz');save_boundaries(output/'partial_snapshots.npz')
        if data is not None:
            failed=sample();failed['integration']=capture_state()
            np.savez_compressed(output/'failure_state.npz',**failed)
        write(output/'failure.json',dict(passed=False,exception=repr(error),traceback=traceback.format_exc(),
            current_control=current_control,current_substep=current_substep,attempted_native_steps=attempts,
            completed_native_steps=completed,verified_native_steps=verified,completed_controls=controls,
            captured_snapshots=len(boundaries),requested_native_steps=12680,requested_controls=1268,
            failed_attempt_has_unclassified_native_state=attempts>completed,new_inference_calls=0,new_labels=0,new_perturbations=0))
        raise


if __name__=='__main__':main()
