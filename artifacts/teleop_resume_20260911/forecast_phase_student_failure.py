"""Fixed private forecasts at every saved learned-control state; no controller run."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-repo', type=Path, required=True)
    parser.add_argument('--oracle', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--physical-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.frozen_repo))
    import mujoco
    import numpy as np
    from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle, sha256
    spec = importlib.util.spec_from_file_location('independent_oracle', args.oracle)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    audit = json.loads(args.physical_audit.read_text())
    assert sha256(args.trace) in audit['input_hashes'].values()
    assert audit['compared_physics_steps'] == 2784
    assert all(audit['original_trace_comparison'].values())
    assert not audit['independent_segment_pass']
    with np.load(args.trace, allow_pickle=False) as archive:
        trace = {k: archive[k].copy() for k in archive.files}
    assert trace['target'].shape == (279, 23)
    assert not np.any(trace['physics_warning_counts'])
    native, contract, *_ = load_native_bundle(args.bundle, 'walk003')
    assert mujoco.__version__ == '3.2.3'
    limits = np.asarray(contract['joint_limits'])
    state_spec = mujoco.mjtState(int(trace['integration_state_spec']))
    args.output.mkdir(exist_ok=False)
    (args.output / 'source.py').write_bytes(Path(__file__).read_bytes())
    inputs = {str(p): sha256(p) for p in (args.trace, args.physical_audit, args.oracle, Path(__file__),
               args.bundle / 'native_prepared.xml', args.bundle / 'prepared_model_arrays.npz', args.bundle / 'contract.json')}
    request = dict(kind='fixed_saved_state_private_forecast', controls=list(range(250, 279)),
                   candidates=['recorded_student', 'recorded_original_bfm_clipped', 'previous_actual_target', 'current_joint_pose'],
                   constant_target_horizons_controls=[1, 5], inputs=inputs,
                   no_connected_controller=True, no_candidate_selected_for_execution=True,
                   no_new_fit_or_labels=True, timing_is_diagnostic_only=True)
    (args.output / 'request.json').write_text(json.dumps(request, indent=2) + '\n')
    rows, timings, exact = [], [], []
    for control in range(250, 279):
        data = mujoco.MjData(native)
        state = trace['control_integration_before'][control]
        mujoco.mj_setState(native, data, state, state_spec)
        mujoco.mj_forward(native, data)
        mujoco.mj_setState(native, data, state, state_spec)
        roundtrip = np.empty(mujoco.mj_stateSize(native, state_spec))
        mujoco.mj_getState(native, data, roundtrip, state_spec)
        np.testing.assert_array_equal(roundtrip, state)
        start = control * 10
        np.testing.assert_array_equal(data.qpos, trace['physics_qpos'][start])
        np.testing.assert_array_equal(data.qvel, trace['physics_qvel'][start])
        candidates = {
            'recorded_student': trace['target'][control],
            'recorded_original_bfm_clipped': np.clip(trace['base_target'][control], limits[:, 0], limits[:, 1]),
            'previous_actual_target': trace['target'][control - 1],
            'current_joint_pose': np.clip(data.qpos[7:], limits[:, 0], limits[:, 1]),
        }
        for name, target in candidates.items():
            for horizon in (1, 5):
                tick = time.perf_counter()
                report, predicted = oracle.inspect_native_segment(native, data, np.tile(target, (horizon, 1)),
                                                                    contract, stop_on_failure=True, retain_trace=True)
                elapsed = (time.perf_counter() - tick) * 1000
                timings.append(dict(control=control, candidate=name, horizon=horizon, milliseconds=elapsed))
                rows.append(dict(control=control, candidate=name, constant_target_controls=horizon,
                                 feasible=report['feasible'], first_failure=report['first_failure'],
                                 maximum=report['maximum'], checked_steps=report['physics_steps'],
                                 source_state_unchanged=report['original_data_unchanged']))
                if name == 'recorded_student' and horizon == 1:
                    count = min(10, 2784 - start)
                    assert report['physics_steps'] == count
                    for predicted_key, saved_key, interval in (
                        ('physics_qpos', 'physics_qpos', slice(start, start + count + 1)),
                        ('physics_qvel', 'physics_qvel', slice(start, start + count + 1)),
                        ('physics_time', 'physics_time', slice(start, start + count + 1)),
                        ('physics_torque', 'physics_torque', slice(start, start + count)),
                        ('physics_actuator_force', 'physics_actuator_torque', slice(start, start + count)),
                    ):
                        np.testing.assert_array_equal(predicted[predicted_key], trace[saved_key][interval])
                    exact.append(dict(control=control, physics_steps=count, all_recorded_samples_bitexact=True))
        print(json.dumps({'control': control, 'counterfactual_forecasts': len(rows)}), flush=True)
    assert len(rows) == 232 and len(exact) == 29
    assert all(sha256(Path(p)) == digest for p, digest in inputs.items())
    timing_summary = {}
    for horizon in (1, 5):
        values = [row['milliseconds'] for row in timings if row['horizon'] == horizon]
        timing_summary[str(horizon)] = np.percentile(values, [50, 95, 100]).tolist()
    result = dict(**request, all_forecasts=rows, actual_student_one_control_reproductions=exact,
                  all_source_integration_states_unchanged=True, timing_ms_p50_p95_max=timing_summary,
                  timings=timings, feasibility_is_only_for_declared_constant_target_segments=True,
                  recursive_safety_or_tracking_qualified=False, hardware_authorized=False)
    (args.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'completed_forecasts': len(rows), 'exact_actual_controls': len(exact),
                      'diagnostic_timing_ms': timing_summary}), flush=True)


if __name__ == '__main__':
    main()
