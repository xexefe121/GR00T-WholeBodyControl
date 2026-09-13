"""Independent positive/negative native323 witnesses for the private oracle."""

import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path('/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
OLD = Path('/mnt/e/codex-artifacts/sonic23_teleop_six_hour_20260910')
NEW = Path('/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911')
sys.path.insert(0, str(NEW / 'preserved_walk_demo_v1/repo'))
import mujoco
import numpy as np
from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle, sha256

ORACLE = ROOT / 'gear_sonic/utils/g1_true23_feasibility_referee.py'
spec = importlib.util.spec_from_file_location('independent_referee', ORACLE)
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)
BUNDLE = ROOT / 'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'
PROBE = NEW / 'recovery_probe_v1'


def archive(path):
    with np.load(path, allow_pickle=False) as a:
        return {k: a[k].copy() for k in a.files}


def restore(native, fixture):
    data = mujoco.MjData(native)
    state_spec = mujoco.mjtState(int(fixture['state_spec']))
    mujoco.mj_setState(native, data, fixture['state_vector'], state_spec)
    mujoco.mj_forward(native, data)
    # Forward fills derived fields but changes warmstart. This is initialization
    # only; reapply the exact integration state before any physical step.
    mujoco.mj_setState(native, data, fixture['state_vector'], state_spec)
    np.testing.assert_array_equal(data.qpos, fixture['qpos'])
    np.testing.assert_array_equal(data.qvel, fixture['qvel'])
    np.testing.assert_array_equal(data.qacc_warmstart, fixture['qacc_warmstart'])
    return data


def run():
    assert mujoco.__version__ == '3.2.3'
    output = NEW / 'independent_feasibility_referee_v1'
    output.mkdir(exist_ok=False)
    cases = {}
    walk_path = OLD / 'mjbatch_full_v1/walk008_v4_native323_allmargin_relativefoot_full_v1/trace.npz'
    walk = archive(walk_path)
    native, contract, *_ = load_native_bundle(BUNDLE, 'walk008')
    data = mujoco.MjData(native)
    data.qpos[:], data.qvel[:] = walk['qpos'][0], walk['qvel'][0]
    mujoco.mj_forward(native, data)
    report, trace = oracle.inspect_native_segment(native, data, walk['target'], contract)
    assert report['feasible'] and report['physics_steps'] == 11140
    for key in ('physics_qpos', 'physics_qvel', 'physics_torque'):
        np.testing.assert_array_equal(trace[key], walk[key])
    report['every_qpos_qvel_torque_bitexact_to_original_trace'] = True
    np.savez_compressed(output / 'complete_walk008.npz', **trace)
    cases['complete_walk008'] = report
    print(json.dumps(dict(case='complete_walk008', report=report)), flush=True)

    native, contract, *_ = load_native_bundle(BUNDLE, 'pico')
    fixture_path = PROBE / 'actual_3740_integration_state.npz'
    probe_path = PROBE / '3740_source_goals_yaw2.npz'
    fixture, probe = archive(fixture_path), archive(probe_path)
    data = restore(native, fixture)
    report, trace = oracle.inspect_native_segment(native, data, probe['target'], contract)
    assert report['feasible'] and report['physics_steps'] == 300
    for key in ('physics_qpos', 'physics_qvel', 'physics_torque'):
        np.testing.assert_array_equal(trace[key], probe[key])
    report['every_qpos_qvel_torque_bitexact_to_recovery_probe'] = True
    np.savez_compressed(output / 'pico3740_safe_BFM.npz', **trace)
    cases['pico3740_safe_BFM'] = report
    print(json.dumps(dict(case='pico3740_safe_BFM', report=report)), flush=True)

    pico_path = OLD / 'mjbatch_full_v1/pico_v4_native323_freshseed_allmargin_5iter_full_v1/trace.npz'
    pico = archive(pico_path)
    kp, kd, effort = [np.asarray(contract[k]) for k in ('kp', 'kd', 'native_effort')]
    data = restore(native, fixture)
    for control in range(3740, 3758):
        target = pico['target'][control]
        for sub in range(10):
            data.ctrl[:] = np.clip(kp * (target - data.qpos[7:]) - kd * data.qvel[6:], -effort, effort)
            mujoco.mj_step(native, data)
            step = control * 10 + sub + 1
            np.testing.assert_array_equal(data.qpos, pico['physics_qpos'][step])
            np.testing.assert_array_equal(data.qvel, pico['physics_qvel'][step])
            np.testing.assert_array_equal(data.ctrl, pico['physics_torque'][step - 1])
    for diagnostic in (False, True):
        name = 'pico3758_diagnostic_full_control' if diagnostic else 'pico3758_first_violation'
        report, trace = oracle.inspect_native_segment(native, data, pico['target'][3758:3759], contract,
                                                       stop_on_failure=not diagnostic)
        assert not report['feasible']
        assert report['first_failure']['reasons'] == ['native_joint_bound']
        assert report['first_failure']['physics_step'] == 3
        assert report['first_failure']['worst_joint_index'] == 4
        assert report['physics_steps'] == (10 if diagnostic else 3)
        known_steps = min(report['physics_steps'], 5)
        for key in ('physics_qpos', 'physics_qvel'):
            np.testing.assert_array_equal(trace[key][:known_steps + 1], pico[key][37580:37580 + known_steps + 1])
        np.testing.assert_array_equal(trace['physics_torque'][:known_steps], pico['physics_torque'][37580:37580 + known_steps])
        report['known_prefix_bitexact_to_original_failure'] = True
        np.savez_compressed(output / (name + '.npz'), **trace)
        cases[name] = report
        print(json.dumps(dict(case=name, report=report)), flush=True)

    report = dict(mujoco=mujoco.__version__, numpy=np.__version__, all_assertions_passed=True,
                  independent_of_production_feasibility_code=True, cases=cases,
                  input_hashes={str(p): sha256(p) for p in (ORACLE, Path(__file__), walk_path,
                      pico_path, fixture_path, probe_path, BUNDLE / 'contract.json',
                      BUNDLE / 'native_prepared.xml', BUNDLE / 'prepared_model_arrays.npz')},
                  full_teleoperation_qualified=False, live_or_hardware_qualified=False)
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (output / 'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (output / 'oracle_snapshot.py').write_bytes(ORACLE.read_bytes())
    print(json.dumps(dict(all_assertions_passed=True, output=str(output))), flush=True)


if __name__ == '__main__':
    run()
