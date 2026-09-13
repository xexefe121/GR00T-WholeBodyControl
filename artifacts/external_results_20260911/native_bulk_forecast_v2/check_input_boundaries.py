"""No-dynamics input/initial-boundary checks with native entry replaced by a spy."""
import hashlib
import json
from pathlib import Path
import sys
import importlib.util
import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = Path('/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
sys.path.insert(0, str(BASE / 'preserved_walk_demo_v1/repo'))
from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle
from bulk_forecast import BulkForecast


class EnteredNativeLoop(Exception):
    pass


def main():
    model, contract, *_ = load_native_bundle(ROOT / 'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1', 'walk003')
    engine = BulkForecast(model, contract)
    with np.load(BASE / 'phase_student_preallocated_forecast_v2/cases.npz', allow_pickle=False) as a:
        state, target = a['integration'][0].copy(), a['target'][0].copy()
    rows = []
    entered = 0
    def spy(*args):
        nonlocal entered
        entered += 1
        raise EnteredNativeLoop('no native step executed')
    engine.loop = spy
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    before, after = np.empty(291), np.empty(291)

    def check(name, change, expected, reason=None):
        source = mujoco.MjData(model)
        mujoco.mj_setState(model, source, state, state_spec)
        mujoco.mj_forward(model, source)
        mujoco.mj_setState(model, source, state, state_spec)
        proposed, horizon = change(source, target.copy(), 5)
        mujoco.mj_getState(model, source, before, state_spec)
        warnings, infos = source.warning.number.copy(), source.warning.lastinfo.copy()
        initial_entered = entered
        try:
            report, trace = engine.predict(source, proposed, horizon)
            outcome = 'initial_failure'
            assert report['physics_steps'] == report['computed_steps'] == report['checked_prefix_steps'] == 0
            assert report['first_failure']['physics_step'] == 0 and not report['feasible']
            assert len(trace['physics_qpos']) == 1
            assert reason in report['first_failure']['reasons']
        except EnteredNativeLoop:
            outcome = 'initial_gates_passed'
        except (AssertionError, ValueError) as error:
            outcome = 'input_rejected'
            report = dict(exception=type(error).__name__, message=str(error))
        assert outcome == expected, (name, outcome, expected)
        assert entered-initial_entered == (1 if expected=='initial_gates_passed' else 0)
        mujoco.mj_getState(model, source, after, state_spec)
        assert before.tobytes() == after.tobytes()
        assert warnings.tobytes() == source.warning.number.tobytes()
        assert infos.tobytes() == source.warning.lastinfo.tobytes()
        assert engine._busy is False and isinstance(engine.private, mujoco.MjData)
        rows.append(dict(name=name, outcome=outcome, no_steps=True, caller291_and_warnings_bitexact=True))

    check('baseline', lambda d,t,h:(t,h), 'initial_gates_passed')
    for h in (0,6,True,5.0):
        check('invalid_horizon_'+repr(h), lambda d,t,_h,h=h:(t,h), 'input_rejected')
    for name, value in [('nan',np.nan),('inf',np.inf),('upper',model.jnt_range[1,1]+.01)]:
        def change(d,t,h,value=value):
            t[0]=value
            return t,h
        check('target_'+name, change, 'input_rejected')
    check('target_float32', lambda d,t,h:(t.astype(np.float32),h), 'input_rejected')
    check('target_noncontiguous', lambda d,t,h:(np.repeat(t,2)[::2],h), 'input_rejected')
    for attribute in ('qfrc_applied','xfrc_applied'):
        def force(d,t,h,attribute=attribute):
            getattr(d,attribute).flat[0]=1.
            return t,h
        check(attribute,force,'input_rejected')
    for delta, expected in ((.9e-6,'initial_gates_passed'),(1.1e-6,'initial_failure')):
        def position(d,t,h,delta=delta):
            d.qpos[7]=model.jnt_range[1,1]+delta
            return t,h
        check('range_'+repr(delta),position,expected,'native_joint_bound')
    for scale, expected in ((1.,'initial_gates_passed'),(1.+1e-12,'initial_failure')):
        def speed(d,t,h,scale=scale):
            d.qvel[6]=contract['native_velocity'][0]*scale
            return t,h
        check('speed_'+repr(scale),speed,expected,'native_joint_speed')
    for height, expected in ((.25,'initial_gates_passed'),(.25-1e-12,'initial_failure')):
        def height_change(d,t,h,height=height):
            d.qpos[2]=height
            return t,h
        check('height_'+repr(height),height_change,expected,'fall')
    for angle, expected in ((1.2-1e-10,'initial_gates_passed'),(1.2+1e-10,'initial_failure')):
        def tilt(d,t,h,angle=angle):
            d.qpos[3:7]=[np.cos(angle/2),np.sin(angle/2),0.,0.]
            return t,h
        check('tilt_'+repr(angle),tilt,expected,'fall')
    for excess, expected in ((.9e-10,'initial_gates_passed'),(1.1e-10,'initial_failure')):
        def quaternion(d,t,h,excess=excess):
            d.qpos[3:7]=[1.+excess,0.,0.,0.]
            return t,h
        check('quaternion_'+repr(excess),quaternion,expected,'invalid_root_quaternion')
    def warning(d,t,h):
        d.warning.number[0]=1
        d.warning.lastinfo[0]=23
        return t,h
    check('existing_warning',warning,'initial_failure','engine_warning')
    result = dict(kind='input_and_initial_boundary_checks_without_dynamics', passed=len(rows),
                  actual_native_calls=0, native_entry_spy_calls=entered, rows=rows,
                  request_sha256=hashlib.sha256((HERE/'request.json').read_bytes()).hexdigest(),
                  test_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out=HERE/'input_boundary_report.json'
    assert not out.exists()
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=len(rows),actual_native_calls=0,native_entry_spy_calls=entered)))


if __name__ == '__main__':
    main()
