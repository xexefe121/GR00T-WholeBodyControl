"""Recover all precontrol integration states by exact audited-command replay."""
import argparse
import json
from pathlib import Path
import sys


def main(args):
    sys.path.insert(0,str(args.frozen_repo))
    import mujoco
    import numpy as np
    from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle, sha256

    assert mujoco.__version__=='3.2.3'
    physical=json.loads(args.physical_audit.read_text())
    digest=sha256(args.trace)
    assert physical['independent_segment_pass'] and physical['recorded_trace_reproduced_through_last_sample']
    assert digest in physical['input_hashes'].values()
    assert sha256(args.fixture) in physical['input_hashes'].values()
    endpoint_report=json.loads((args.expected_endpoint.parent/'report.json').read_text())
    assert endpoint_report['kind']=='independent_native_full_integration_endpoint_reconstruction'
    assert endpoint_report['original_trace_sha256']==digest
    assert endpoint_report['endpoint_sha256']==sha256(args.expected_endpoint)
    native,contract,*_=load_native_bundle(args.bundle,args.clip)
    with np.load(args.fixture,allow_pickle=False) as a:
        initial=a['state_vector'].copy()
        spec=int(a['state_spec'])
    assert spec==int(mujoco.mjtState.mjSTATE_INTEGRATION) and mujoco.mj_stateSize(native,spec)==291
    with np.load(args.trace,allow_pickle=False) as a:
        fields=('target','physics_qpos','physics_qvel','physics_torque','physics_actuator_force',
                'physics_time','physics_warning_number','physics_warning_lastinfo','physics_substeps')
        trace={key:a[key].copy() for key in fields}
    # Canonical fixtures contain integration fields only. The already audited
    # original trace binds the separate initial warning ledger exactly.
    assert not np.any(trace['physics_warning_number'][0])
    assert not np.any(trace['physics_warning_lastinfo'][0])
    count=len(trace['target'])
    assert count==physical['intended_segment_controls']==endpoint_report['completed_controls']
    assert np.array_equal(trace['physics_substeps'],np.full(count,10))
    args.output.mkdir(parents=True,exist_ok=False)
    paths=[args.trace,args.fixture,args.physical_audit,args.expected_endpoint,
           args.expected_endpoint.parent/'report.json',args.bundle/'contract.json',
           args.bundle/'native_prepared.xml',args.bundle/'prepared_model_arrays.npz',Path(__file__),
           args.frozen_repo/'gear_sonic/utils/g1_true23_mjbatch_mpc.py']
    request=dict(kind='full_audited_recorded_command_replay_for_all_precontrol_states',clip=args.clip,
                 full_controls=count,requested_physics_steps=count*10,requested_snapshots=count,
                 new_controller_inference=False,inputs={str(p):sha256(p) for p in paths})
    (args.output/'request.json').write_text(json.dumps(request,indent=2)+'\n')
    (args.output/'source.py').write_bytes(Path(__file__).read_bytes())
    data=mujoco.MjData(native)
    mujoco.mj_setState(native,data,initial,spec)
    mujoco.mj_forward(native,data)
    mujoco.mj_setState(native,data,initial,spec)
    expected=data.time
    kp,kd,effort=[np.asarray(contract[k]) for k in ('kp','kd','native_effort')]
    snapshots=np.empty((count,291))
    times=np.empty(count)
    warnings=np.empty((count,8),dtype=np.int32)
    infos=np.empty_like(warnings)
    completed=step=0

    def exact(a,b,key):
        a,b=np.asarray(a),np.asarray(b)
        assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),(step,key)

    def compare():
        exact(data.qpos,trace['physics_qpos'][step],'qpos')
        exact(data.qvel,trace['physics_qvel'][step],'qvel')
        exact(data.warning.number,trace['physics_warning_number'][step],'warning counts')
        exact(data.warning.lastinfo,trace['physics_warning_lastinfo'][step],'warning lastinfo')
        assert data.time==trace['physics_time'][step]==expected
        if step:
            exact(data.ctrl,trace['physics_torque'][step-1],'command torque')
            exact(data.qfrc_actuator[6:],trace['physics_actuator_force'][step-1],'actuator force')

    try:
        compare()
        for control,target in enumerate(trace['target']):
            assert not np.any(data.qfrc_applied) and not np.any(data.xfrc_applied)
            mujoco.mj_getState(native,data,snapshots[control],spec)
            times[control]=data.time
            warnings[control]=data.warning.number
            infos[control]=data.warning.lastinfo
            for _ in range(10):
                data.ctrl[:]=np.minimum(np.maximum(kp*(target-data.qpos[7:])-kd*data.qvel[6:],-effort),effort)
                mujoco.mj_step(native,data)
                expected+=.002
                step+=1
                compare()
            completed=control+1
            if completed%500==0:
                print(json.dumps(dict(clip=args.clip,completed_controls=completed,compared_steps=step)),flush=True)
        assert step==count*10
        final=np.empty(291)
        mujoco.mj_getState(native,data,final,spec)
        with np.load(args.expected_endpoint,allow_pickle=False) as a:
            exact(final,a['final_integration'],'independent final291')
            exact(data.qacc_warmstart,a['qacc_warmstart'],'independent final warmstart')
            exact(data.ctrl,a['ctrl'],'independent final ctrl')
            assert data.time==float(a['time'])
    except Exception as error:
        np.savez_compressed(args.output/'partial_snapshots.npz',control=np.arange(completed),
                            control_integration_before=snapshots[:completed],time=times[:completed])
        (args.output/'failure.json').write_text(json.dumps(dict(completed_controls=completed,compared_steps=step,
                                  exception=type(error).__name__,message=str(error)),indent=2)+'\n')
        raise
    path=args.output/'control_snapshots.npz'
    np.savez_compressed(path,control=np.arange(count,dtype=np.int64),control_integration_before=snapshots,
                        integration_spec=np.asarray(spec),time=times,
                        qpos=trace['physics_qpos'][::10][:-1],qvel=trace['physics_qvel'][::10][:-1],
                        warning_counts=warnings,warning_lastinfo=infos,
                        final_integration=final,original_trace_sha256=np.asarray(digest))
    for p,expected_hash in request['inputs'].items():
        assert sha256(p)==expected_hash,p
    report=dict(kind='independent_all_precontrol_native_integration_reconstruction',clip=args.clip,
                full_controls=count,precontrol_snapshots=count,integration_state_size=291,
                compared_physics_steps=step,all_recorded_samples_byteexact=True,
                final291_matches_previously_reconstructed_endpoint_byteexact=True,
                independent_physics_already_passed=True,original_trace_sha256=digest,
                snapshots_sha256=sha256(path),request_sha256=sha256(args.output/'request.json'),
                source_state_rewrites_after_initialization=0,root_forces=False,
                new_controller_inference=False,source_features_or_labels_generated=False,
                hashes=request['inputs'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='hashes'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('frozen-repo','bundle','fixture','trace','physical-audit','expected-endpoint','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--clip',required=True,choices=('pico','walk002','walk003','walk008'))
    main(parser.parse_args())
