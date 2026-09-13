"""Two fixed saved-case alias checks and no-dynamics entry-lock checks."""
import hashlib
import json
from pathlib import Path
import sys
import threading
import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = Path('/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
sys.path.insert(0,str(BASE/'preserved_walk_demo_v1/repo'))
from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle
from bulk_forecast import BulkForecast


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    out=HERE/'ownership_checks'
    out.mkdir(exist_ok=False)
    paths=[Path(__file__),HERE/'bulk_forecast.py',HERE/'request.json',HERE/'results/report.json',
           HERE/'results/oracle_traces.npz',BASE/'phase_student_preallocated_forecast_v2/cases.npz']
    request=dict(kind='fixed_source_case0_two_target_alias_representations_and_entry_lock_checks',
                 inputs={str(p):sha(p) for p in paths},case=0,alias_types=['trace_buffer','private_qpos'],
                 actor_calls=0,actual_plant_steps=0,extra_private_forecast_calls=2)
    (out/'request.json').write_text(json.dumps(request,indent=2)+'\n')
    model,contract,*_=load_native_bundle(ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1','walk003')
    engine=BulkForecast(model,contract)
    source=mujoco.MjData(model)
    spec=mujoco.mjtState.mjSTATE_INTEGRATION
    with np.load(BASE/'phase_student_preallocated_forecast_v2/cases.npz',allow_pickle=False) as a:
        state,target,controls=a['integration'][0].copy(),a['target'][0].copy(),int(a['horizon_controls'][0])
    before,after=np.empty(291),np.empty(291)
    rows=[]
    with np.load(HERE/'results/oracle_traces.npz',allow_pickle=False) as old:
        steps=int(old['computed_steps'][0])
        for name in request['alias_types']:
            mujoco.mj_setState(model,source,state,spec)
            mujoco.mj_forward(model,source)
            mujoco.mj_setState(model,source,state,spec)
            mujoco.mj_getState(model,source,before,spec)
            view=engine.q[0,7:] if name=='trace_buffer' else engine.private.qpos[7:]
            np.copyto(view,target)
            report,trace=engine.predict(source,view,controls)
            assert report['computed_steps']==steps
            for key,a in trace.items():
                expected=old[key][0,:len(a)]
                assert a.dtype==expected.dtype and a.tobytes()==expected.tobytes(),(name,key)
            assert engine.owned_target.tobytes()==target.tobytes()
            mujoco.mj_getState(model,source,after,spec)
            assert before.tobytes()==after.tobytes()
            rows.append(dict(alias_type=name,private_steps=steps,all_seven_fields_bitexact=True,caller_unchanged=True))
    errors=[]
    engine._call_lock.acquire()
    try:
        def worker():
            try:
                # No valid source/target: lock must reject before reading either.
                engine.predict(object(),object(),object())
            except RuntimeError as error:
                errors.append(str(error))
        thread=threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive() and errors==['BulkForecast is nonconcurrent']
    finally:
        engine._call_lock.release()
    original_loop=engine.loop
    def nested_spy(*args):
        try:
            engine.predict(object(),object(),object())
        except RuntimeError as error:
            assert str(error)=='BulkForecast is nonconcurrent'
        else:
            raise AssertionError('nested call was not rejected')
        raise LookupError('injected native entry failure')
    engine.loop=nested_spy
    try:
        engine.predict(source,target,controls)
    except LookupError as error:
        assert str(error)=='injected native entry failure'
    else:
        raise AssertionError('injected entry failure did not propagate')
    finally:
        engine.loop=original_loop
    assert not engine._busy and engine._call_lock.acquire(blocking=False)
    engine._call_lock.release()
    mujoco.mj_getState(model,source,after,spec)
    assert before.tobytes()==after.tobytes()
    for p,expected in request['inputs'].items():
        assert sha(p)==expected
    result=dict(passed=True,alias_rows=rows,other_thread_rejected_before_inputs=True,
                nested_call_rejected=True,exception_releases_lock=True,source_unchanged=True,
                total_private_steps=sum(r['private_steps'] for r in rows),actual_plant_steps=0,
                request_sha256=sha(out/'request.json'))
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
