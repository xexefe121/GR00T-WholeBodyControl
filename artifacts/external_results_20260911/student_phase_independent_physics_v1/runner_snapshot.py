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
    if args.requested_controls <= 0:
        raise ValueError('requested-controls must declare a positive intended segment length')
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
        numeric_fields = {'actual_targets', 'target', 'physics_states', 'physics_qpos',
                          'physics_qvel', 'physics_torque', 'physics_torques',
                          'physics_actuator_force', 'physics_actuator_torque', 'physics_time',
                          'physics_warning_number', 'physics_warning_counts', 'physics_warning_lastinfo',
                          'physics_substeps'}
        expected = {key: archive[key].copy() for key in archive.files if key in numeric_fields}
    producer_report_path = args.trace.parent / 'report.json'
    producer = None
    if producer_report_path.is_file():
        producer = json.loads(producer_report_path.read_text())
        if 'trace_sha256' not in producer:
            matched = [case for case in producer.get('cases', [])
                       if case.get('trace_sha256') == sha256(args.trace)]
            if len(matched) != 1:
                raise ValueError('multi-case report must contain exactly one matching trace hash')
            producer = matched[0]
        assert producer['trace_sha256'] == sha256(args.trace)
        if 'requested_controls' in producer:
            assert producer['requested_controls'] == args.requested_controls
    targets = expected['actual_targets'] if 'actual_targets' in expected else expected['target']
    if 'physics_states' in expected:
        qpos, qvel = expected['physics_states'][:, :30], expected['physics_states'][:, 30:]
    else:
        qpos, qvel = expected['physics_qpos'], expected['physics_qvel']
    assert targets.ndim == 2 and targets.shape[1] == 23 and len(targets) > 0
    recorded_steps = len(qpos) - 1
    assert qpos.shape == (recorded_steps + 1, 30)
    assert qvel.shape == (recorded_steps + 1, 29)
    assert (len(targets) - 1) * 10 < recorded_steps <= len(targets) * 10
    if 'physics_substeps' in expected:
        substeps = expected['physics_substeps']
        assert substeps.shape == (len(targets),)
        assert np.array_equal(substeps, substeps.astype(int))
        assert np.all(substeps[:-1] == 10) and 1 <= substeps[-1] <= 10
        assert int(substeps.sum()) == recorded_steps
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
    # A strict replay may stop before a producer using weaker historical gates.
    # Conversely a partial producer trace must never acquire unrecorded samples
    # merely because its last target could be replayed for a full control.
    count = min(report['physics_steps'], recorded_steps)
    comparisons = dict(
        all_qpos_bitexact=bool(np.array_equal(trace['physics_qpos'][:count + 1], qpos[:count + 1])),
        all_qvel_bitexact=bool(np.array_equal(trace['physics_qvel'][:count + 1], qvel[:count + 1])),
    )
    if 'physics_torque' in expected:
        commands = expected['physics_torque']
        assert commands.shape == (recorded_steps, 23)
        comparisons['all_command_torque_bitexact'] = bool(np.array_equal(trace['physics_torque'][:count], commands[:count]))
    if 'physics_torques' in expected:
        # Bounded probe producer stores actual generalized actuator force here.
        forces = expected['physics_torques']
        assert forces.shape == (recorded_steps, 23)
        comparisons['all_actuator_torque_bitexact'] = bool(np.array_equal(trace['physics_actuator_force'][:count], forces[:count]))
    for force_key in ('physics_actuator_force', 'physics_actuator_torque'):
        if force_key in expected:
            forces = expected[force_key]
            assert forces.shape == (recorded_steps, 23)
            comparisons[force_key + '_bitexact'] = bool(np.array_equal(trace['physics_actuator_force'][:count], forces[:count]))
    if 'physics_time' in expected:
        times = expected['physics_time']
        assert times.shape == (recorded_steps + 1,)
        comparisons['all_physics_time_bitexact'] = bool(np.array_equal(trace['physics_time'][:count + 1], times[:count + 1]))
    for actual_key, expected_key in (('warning_counts', 'physics_warning_number'),
                                     ('warning_counts', 'physics_warning_counts'),
                                     ('warning_lastinfo', 'physics_warning_lastinfo')):
        if expected_key in expected:
            values = expected[expected_key]
            assert values.shape == (recorded_steps + 1, 8)
            assert np.issubdtype(values.dtype, np.integer)
            comparisons[expected_key + '_bitexact'] = bool(np.array_equal(trace[actual_key][:count + 1], values[:count + 1]))
    witnesses = []
    failure = report['first_failure']
    if failure is not None:
        failure['worst_joint_index_scope'] = 'legacy maximum joint range excess; see independent_joint_witnesses for speed or effort'
        step = failure['physics_step']
        limits = np.asarray(contract['joint_limits'])
        positions = trace['physics_qpos'][step, 7:]
        ratios = np.abs(trace['physics_qvel'][step, 6:]) / np.asarray(contract['native_velocity'])
        for reason, values in (
            ('native_joint_bound', np.maximum(limits[:, 0] - positions, positions - limits[:, 1])),
            ('native_joint_speed', ratios),
        ):
            if reason in failure['reasons']:
                joint = int(np.argmax(values))
                witnesses.append(dict(reason=reason, joint_index=joint, joint_name=contract['joint_names'][joint],
                    qpos_rad=float(positions[joint]), lower_rad=float(limits[joint, 0]), upper_rad=float(limits[joint, 1]),
                    qvel_radps=float(trace['physics_qvel'][step, 6 + joint]),
                    native_velocity_radps=float(contract['native_velocity'][joint]),
                    constraint_value=float(values[joint])))
    report.update(
        evaluated_saved_controls=len(targets),
        recorded_physics_steps=recorded_steps,
        recorded_full_controls=recorded_steps // 10,
        recorded_partial_substeps=recorded_steps % 10,
        compared_physics_steps=count,
        recorded_trace_reproduced_through_last_sample=count == recorded_steps and all(comparisons.values()),
        private_replay_steps_beyond_recorded_prefix=max(0, report['physics_steps'] - recorded_steps),
        independent_joint_witnesses=witnesses,
        intended_segment_controls=args.requested_controls,
        requested_segment_completed=count == args.requested_controls * 10 and len(targets) == args.requested_controls,
        original_trace_comparison=comparisons,
        independent_segment_pass=report['feasible'] and count == args.requested_controls * 10
            and len(targets) == args.requested_controls and all(comparisons.values()),
        restored_state_diagnostic=True, full_source_qualified=False,
        feedback_replanning_reproduced=False,
        producer_completion=None if producer is None else dict(
            completed=producer.get('completed', producer.get('full_horizon_physical_pass')), failure=producer.get('failure'),
            requested_controls=producer.get('requested_controls'),
            completed_controls=producer.get('completed_controls', producer.get('executed_controls', producer.get('full_controls')))),
        initialization='full actual mjSTATE_INTEGRATION fixture; forward then restore again before the first step',
        input_hashes={str(path): sha256(path) for path in (
            args.fixture, args.trace, args.oracle, Path(__file__), args.bundle / 'contract.json',
            args.bundle / 'native_prepared.xml', args.bundle / 'prepared_model_arrays.npz',
            args.frozen_repo / 'gear_sonic/utils/g1_true23_mjbatch_mpc.py')},
    )
    if producer is not None:
        report['input_hashes'][str(producer_report_path)] = sha256(producer_report_path)
    np.savez_compressed(args.output / 'trace.npz', **trace)
    report['trace_sha256'] = sha256(args.output / 'trace.npz')
    (args.output / 'oracle_snapshot.py').write_bytes(args.oracle.read_bytes())
    (args.output / 'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: report[key] for key in (
        'independent_segment_pass', 'evaluated_saved_controls', 'intended_segment_controls',
        'physics_steps', 'feasible', 'original_trace_comparison', 'maximum', 'producer_completion'
    )}), flush=True)
    return 0 if report['independent_segment_pass'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('frozen-repo', 'oracle', 'bundle', 'fixture', 'trace', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--clip', required=True, choices=('pico', 'walk002', 'walk003', 'walk008'))
    parser.add_argument('--requested-controls', required=True, type=int,
                        help='Original intended diagnostic length, never shortened to fit an aborted trace.')
    raise SystemExit(run(parser.parse_args()))
