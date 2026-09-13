"""Recover full native integration state after an independently qualified trace."""

import argparse
import json
from pathlib import Path
import sys


def main(args):
    sys.path.insert(0, str(args.frozen_repo))
    import mujoco
    import numpy as np
    from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle, sha256

    assert mujoco.__version__ == '3.2.3'
    physical = json.loads(args.physical_audit.read_text())
    digest = sha256(args.trace)
    assert physical['independent_segment_pass']
    assert physical['recorded_trace_reproduced_through_last_sample']
    assert digest in physical['input_hashes'].values()
    assert sha256(args.fixture) in physical['input_hashes'].values()
    native, contract, *_ = load_native_bundle(args.bundle, args.clip)
    with np.load(args.fixture, allow_pickle=False) as fixture:
        state = fixture['state_vector'].copy()
        spec = int(fixture['state_spec'])
    assert spec == int(mujoco.mjtState.mjSTATE_INTEGRATION)
    with np.load(args.trace, allow_pickle=False) as saved:
        fields = ('target', 'physics_qpos', 'physics_qvel', 'physics_torque', 'physics_actuator_force',
                  'physics_time', 'physics_warning_number', 'physics_warning_lastinfo', 'physics_substeps')
        trace = {name: saved[name].copy() for name in fields}
    count = len(trace['target'])
    assert count == physical['intended_segment_controls']
    np.testing.assert_array_equal(trace['physics_substeps'], np.full(count, 10))
    data = mujoco.MjData(native)
    mujoco.mj_setState(native, data, state, spec)
    mujoco.mj_forward(native, data)
    mujoco.mj_setState(native, data, state, spec)
    expected_time = data.time
    kp, kd, effort = [np.asarray(contract[name]) for name in ('kp', 'kd', 'native_effort')]
    assert not np.any(data.qfrc_applied) and not np.any(data.xfrc_applied)

    def compare(step):
        np.testing.assert_array_equal(data.qpos, trace['physics_qpos'][step])
        np.testing.assert_array_equal(data.qvel, trace['physics_qvel'][step])
        np.testing.assert_array_equal(data.warning.number, trace['physics_warning_number'][step])
        np.testing.assert_array_equal(data.warning.lastinfo, trace['physics_warning_lastinfo'][step])
        assert data.time == trace['physics_time'][step] == expected_time
        if step:
            np.testing.assert_array_equal(data.ctrl, trace['physics_torque'][step - 1])
            np.testing.assert_array_equal(data.qfrc_actuator[6:], trace['physics_actuator_force'][step - 1])

    compare(0)
    step = 0
    for target in trace['target']:
        for _ in range(10):
            data.ctrl[:] = np.minimum(np.maximum(kp * (target - data.qpos[7:]) - kd * data.qvel[6:], -effort), effort)
            mujoco.mj_step(native, data)
            expected_time += .002
            step += 1
            compare(step)
    assert step == count * 10 and not np.any(data.qfrc_applied) and not np.any(data.xfrc_applied)
    endpoint = np.empty(mujoco.mj_stateSize(native, spec))
    mujoco.mj_getState(native, data, endpoint, spec)
    args.output.mkdir(parents=True, exist_ok=False)
    endpoint_path = args.output / 'endpoint.npz'
    np.savez_compressed(endpoint_path, final_integration=endpoint, integration_state_spec=spec,
        completed_controls=count, original_trace_sha256=np.asarray(digest), qpos=data.qpos.copy(),
        qvel=data.qvel.copy(), qacc_warmstart=data.qacc_warmstart.copy(), ctrl=data.ctrl.copy(),
        time=np.asarray(data.time), warning_counts=data.warning.number.copy(),
        warning_lastinfo=data.warning.lastinfo.copy())
    report = dict(kind='independent_native_full_integration_endpoint_reconstruction',
        completed_controls=count, compared_physics_steps=step, all_recorded_samples_bitexact=True,
        independent_physical_audit_already_passed=True, original_trace_sha256=digest,
        final_time=float(data.time), integration_state_size=len(endpoint),
        source_state_rewrites_after_initialization=0, root_forces=False,
        new_controller_commands_executed=False, continuation_qualified=False,
        endpoint_sha256=sha256(endpoint_path),
        hashes={str(p): sha256(p) for p in (args.trace, args.fixture, args.physical_audit,
            args.bundle / 'contract.json', args.bundle / 'native_prepared.xml',
            args.bundle / 'prepared_model_arrays.npz', Path(__file__),
            args.frozen_repo / 'gear_sonic/utils/g1_true23_mjbatch_mpc.py')})
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (args.output / 'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({key: value for key, value in report.items() if key != 'hashes'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('frozen-repo', 'bundle', 'fixture', 'trace', 'physical-audit', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--clip', required=True)
    main(parser.parse_args())
