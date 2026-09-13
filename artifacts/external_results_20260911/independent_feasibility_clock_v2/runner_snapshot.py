"""Native engine clock recurrence plus deliberately corrupted clock witnesses."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from verify_feasibility_referee import archive, restore, oracle, BUNDLE, NEW, PROBE, ORACLE
from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle, sha256
import mujoco
import numpy as np

output = NEW / 'independent_feasibility_clock_v2'
output.mkdir(exist_ok=False)
engine = mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".002" gravity="0 0 0"/>
<worldbody><body><freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>''')
data = mujoco.MjData(engine)
expected = 0.
old_false_failure = None
for step in range(1, 65301):
    mujoco.mj_step(engine, data)
    expected += .002
    assert data.time == expected
    assert not np.any(data.warning.number)
    if old_false_failure is None and abs(data.time - step * .002) > 1e-10:
        old_false_failure = step
assert old_false_failure == 60370
engine_clock = dict(physics_steps=65300, independent_recurrence_bitexact=True,
                    old_ideal_time_first_false_failure=old_false_failure,
                    final_ideal_time_difference_seconds=float(data.time - 65300 * .002),
                    model_scope='simple zero-gravity engine clock witness, not native23 controller qualification')

native, contract, *_ = load_native_bundle(BUNDLE, 'pico')
fixture_path = PROBE / 'actual_3740_integration_state.npz'
probe_path = PROBE / '3740_source_goals_yaw2.npz'
fixture, probe = archive(fixture_path), archive(probe_path)
cases = {}
for mode in ('normal', 'reset', 'missing_step', 'extra_step', 'subtolerance_cumulative_drift'):
    data = restore(native, fixture)
    raw_step = mujoco.mj_step
    counter = [0]

    def altered_step(model, state):
        raw_step(model, state)
        counter[0] += 1
        if counter[0] == 3:
            if mode == 'reset': state.time = 0.
            if mode == 'missing_step': state.time -= .002
            if mode == 'extra_step': state.time += .002
        if mode == 'subtolerance_cumulative_drift': state.time += 1e-12

    try:
        if mode != 'normal': mujoco.mj_step = altered_step
        report, trace = oracle.inspect_native_segment(native, data, probe['target'], contract)
    finally:
        mujoco.mj_step = raw_step
    if mode == 'normal':
        assert report['feasible'] and report['physics_steps'] == 300
        assert report['maximum']['clock_error_seconds'] == 0.
        for key in ('physics_qpos', 'physics_qvel', 'physics_torque'):
            np.testing.assert_array_equal(trace[key], probe[key])
    else:
        assert not report['feasible']
        assert report['first_failure']['reasons'] == ['physics_clock']
        if mode == 'subtolerance_cumulative_drift':
            assert 90 <= report['physics_steps'] <= 110
        else:
            assert report['physics_steps'] == 3
    report['deliberately_altered_clock_fixture'] = mode != 'normal'
    cases[mode] = report
    print(json.dumps(dict(case=mode, feasible=report['feasible'], steps=report['physics_steps'])), flush=True)

report = dict(all_assertions_passed=True, mujoco=mujoco.__version__, engine_clock=engine_clock,
              native23_clock_cases=cases, tolerance_seconds=1e-10,
              full_source_or_hardware_qualified=False,
              input_hashes={str(p): sha256(p) for p in (Path(__file__), ORACLE, fixture_path, probe_path)})
(output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
(output / 'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
(output / 'oracle_snapshot.py').write_bytes(ORACLE.read_bytes())
print(json.dumps(dict(all_assertions_passed=True, output=str(output))), flush=True)
