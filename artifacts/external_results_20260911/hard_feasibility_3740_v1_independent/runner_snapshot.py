"""Independently audit a bounded experiment restored from actual integration state.

This audits its explicit executed targets; it does not rerun planning, certify
feedback robustness, or qualify an entire source recording from a restored state.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys


def run(args):
    sys.path.insert(0, str(args.frozen_repo))
    import mujoco
    import numpy as np
    from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle, sha256

    assert mujoco.__version__ == '3.2.3'
    spec = importlib.util.spec_from_file_location('independent_referee', args.oracle)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    args.output.mkdir(parents=True, exist_ok=False)
    native, contract, *_ = load_native_bundle(args.bundle, args.clip)
    with np.load(args.fixture, allow_pickle=False) as archive:
        fixture = {key: archive[key].copy() for key in archive.files}
    with np.load(args.trace, allow_pickle=False) as archive:
        expected = {key: archive[key].copy() for key in archive.files}
    targets = expected['actual_targets'] if 'actual_targets' in expected else expected['target']
    if 'physics_states' in expected:
        qpos, qvel = expected['physics_states'][:, :30], expected['physics_states'][:, 30:]
    else:
        qpos, qvel = expected['physics_qpos'], expected['physics_qvel']
    assert targets.ndim == 2 and targets.shape[1] == 23 and len(targets) > 0
    assert qpos.shape == (len(targets) * 10 + 1, 30)
    assert qvel.shape == (len(targets) * 10 + 1, 29)
    data = mujoco.MjData(native)
    state_spec = mujoco.mjtState(int(fixture['state_spec']))
    mujoco.mj_setState(native, data, fixture['state_vector'], state_spec)
    mujoco.mj_forward(native, data)
    mujoco.mj_setState(native, data, fixture['state_vector'], state_spec)
    np.testing.assert_array_equal(data.qpos, fixture['qpos'])
    np.testing.assert_array_equal(data.qvel, fixture['qvel'])
    np.testing.assert_array_equal(data.qacc_warmstart, fixture['qacc_warmstart'])
    np.testing.assert_array_equal(data.qpos, qpos[0])
    np.testing.assert_array_equal(data.qvel, qvel[0])
    report, trace = oracle.inspect_native_segment(native, data, targets, contract)
    count = report['physics_steps']
    comparisons = dict(
        all_qpos_bitexact=bool(np.array_equal(trace['physics_qpos'], qpos[:count + 1])),
        all_qvel_bitexact=bool(np.array_equal(trace['physics_qvel'], qvel[:count + 1])),
    )
    if 'physics_torque' in expected:
        commands = expected['physics_torque']
        assert commands.shape == (len(targets) * 10, 23)
        comparisons['all_command_torque_bitexact'] = bool(np.array_equal(trace['physics_torque'], commands[:count]))
    if 'physics_torques' in expected:
        # Bounded probe producer stores actual generalized actuator force here.
        forces = expected['physics_torques']
        assert forces.shape == (len(targets) * 10, 23)
        comparisons['all_actuator_torque_bitexact'] = bool(np.array_equal(trace['physics_actuator_force'], forces[:count]))
    if 'physics_time' in expected:
        times = expected['physics_time']
        assert times.shape == (len(targets) * 10 + 1,)
        comparisons['all_physics_time_bitexact'] = bool(np.array_equal(trace['physics_time'], times[:count + 1]))
    report.update(
        requested_segment_completed=count == len(targets) * 10,
        original_trace_comparison=comparisons,
        independent_segment_pass=report['feasible'] and count == len(targets) * 10 and all(comparisons.values()),
        restored_state_diagnostic=True, full_source_qualified=False,
        feedback_replanning_reproduced=False,
        initialization='full actual mjSTATE_INTEGRATION fixture; forward then restore again before the first step',
        input_hashes={str(path): sha256(path) for path in (
            args.fixture, args.trace, args.oracle, Path(__file__), args.bundle / 'contract.json',
            args.bundle / 'native_prepared.xml', args.bundle / 'prepared_model_arrays.npz',
            args.frozen_repo / 'gear_sonic/utils/g1_true23_mjbatch_mpc.py')},
    )
    np.savez_compressed(args.output / 'trace.npz', **trace)
    report['trace_sha256'] = sha256(args.output / 'trace.npz')
    (args.output / 'oracle_snapshot.py').write_bytes(args.oracle.read_bytes())
    (args.output / 'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report), flush=True)
    return 0 if report['independent_segment_pass'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('frozen-repo', 'oracle', 'bundle', 'fixture', 'trace', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--clip', required=True, choices=('pico', 'walk002', 'walk003', 'walk008'))
    raise SystemExit(run(parser.parse_args()))
