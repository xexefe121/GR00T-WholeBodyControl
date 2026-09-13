"""Collect actual expert commands only after explicit full independent qualification.

No optimization, training or physics steps occur here. Imports that create model
or ONNX state are delayed until the independent qualification receipt passes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

BASE = Path(__file__).resolve().parent.parent
TASK = BASE.parent
RUN = TASK / 'bfm_entry250_actual_oracle_v1'
BASELINE = TASK / 'original_bfm_entry250_v1'
PILOT = TASK / 'fast_controller_nominal_pilot_v1'
STUDENT = TASK / 'fast_controller_aggregate_fit_v1'
KEYS = ('actions', 'base_ang_vel', 'dof_pos', 'dof_vel', 'projected_gravity')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local(path):
    value = str(path).replace('\\', '/')
    if sys.platform != 'win32' and len(value) > 2 and value[1] == ':':
        value = '/mnt/' + value[0].lower() + value[2:]
    return Path(value)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def archive(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def check_qualification(path):
    frozen = read(BASE / 'collector_frozen_inputs.json')
    for name, digest in frozen['source_sha256'].items():
        assert sha(Path(__file__).parent / name) == digest, name
    for name, digest in frozen['input_sha256'].items():
        assert sha(local(name)) == digest, name
    assert sha(path) == frozen['qualification_receipt_sha256']
    decision = read(path)
    assert decision['root_authorized_extraction'] is True
    assert decision['model_fitting_authorized'] is False
    nominal = RUN / 'nominal/trace.npz'
    extension = RUN / 'post_lifecycle_hold_5s/trace.npz'
    assert sha(nominal) == decision['nominal_trace_sha256']
    assert sha(extension) == decision['extension_trace_sha256']
    reports = {}
    for key in ('nominal_physics', 'nominal_intent', 'extension_physics', 'extension_intent'):
        item = decision['independent_reports'][key]
        report_path = local(item['path'])
        assert sha(report_path) == item['sha256'], key
        report = read(report_path)
        digest = decision['extension_trace_sha256' if key.startswith('extension') else 'nominal_trace_sha256']
        assert digest in report.get('input_hashes', report.get('hashes', {})).values(), (key, 'trace binding')
        if key.endswith('physics'):
            assert report['independent_segment_pass'] is True
            assert report['recorded_trace_reproduced_through_last_sample'] is True
            assert all(report['original_trace_comparison'].values())
            assert report['intended_segment_controls'] == (250 if key.startswith('extension') else 1569)
        else:
            assert report['requested_segment_quiet_pass'] is True
            assert report['intended_segment_completed'] is True
            assert report['independent_physical_pass'] is True
            assert report['requested_controls'] == (250 if key.startswith('extension') else 1569)
            assert report['global_start'] == (1569 if key.startswith('extension') else 0)
            if key.startswith('nominal'):
                assert report['full_lifecycle_source_intent_pass'] is True
                assert report['source_metrics']['source_controls'] == 819
        reports[key] = report
    assert sha(RUN / 'inputs/baseline250_trace.npz') == frozen['baseline250_sha256']
    assert sha(RUN / 'frozen_inputs.json') == frozen['expert_branch_receipt_sha256']
    assert sha(RUN / 'prelaunch_clearance.json') == frozen['expert_launch_clearance_sha256']
    return decision, frozen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qualification', type=Path, required=True)
    args = parser.parse_args()
    decision, frozen = check_qualification(args.qualification)
    destination = BASE / 'labels'
    destination.mkdir(exist_ok=False)
    # Existing feature/base helpers remain byte-identical to the reviewed pilot.
    import mujoco
    from student_linear_runtime import (LinearFeatures, infer_base, BUNDLE, REFERENCE,
        ONNX, DEPS, Native23BFMRolloutSeed, FEATURES, OFFSETS)
    from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle, load_motion_override
    from gear_sonic.utils.g1_true23_bfm_seed_observations import BFMHistory
    assert mujoco.__version__ == '3.2.3'
    native, c, original, timeline, manifest = load_native_bundle(BUNDLE, 'walk003')
    motion, _ = load_motion_override(REFERENCE, BUNDLE, 'walk003', native, c, original, timeline, manifest)
    original29 = archive(BUNDLE / 'walk003/original29.npz')
    trace = archive(RUN / 'nominal/trace.npz')
    baseline = archive(RUN / 'inputs/baseline250_trace.npz')
    snapshot = archive(RUN / 'inputs/precontrol250.npz')
    extension = archive(RUN / 'post_lifecycle_hold_5s/trace.npz')
    np.testing.assert_array_equal(trace['global_control'], np.arange(1569))
    np.testing.assert_array_equal(trace['source_frame'], np.arange(1569) + 11)
    np.testing.assert_array_equal(trace['physics_substeps'], np.full(1569, 10))
    np.testing.assert_array_equal(trace['controller_mode'], np.r_[np.zeros(250), np.ones(1019), np.full(300, 2)])
    np.testing.assert_array_equal(extension['global_control'], np.arange(1569, 1819))
    np.testing.assert_array_equal(extension['physics_substeps'], np.full(250, 10))
    np.testing.assert_array_equal(trace['final_integration'], extension['initial_integration'])
    np.testing.assert_array_equal(trace['final_previous_action'], extension['control_previous_action_before'][0])
    for key in KEYS:
        np.testing.assert_array_equal(trace['final_history_' + key], extension['control_history_' + key][0])
    direct=('target','source_frame','global_control','controller_mode','joint_error','root_error','physics_substeps',
        'control_integration_before','control_history_before','control_previous_action_before','previous_action','action',
        'qpos','qvel','physics_qpos','physics_qvel','physics_torque','physics_time','physics_expected_time',
        'range_excess','velocity_ratio','effort_ratio','clock_error')
    mapping={key:key for key in direct}
    mapping.update(physics_actuator_force='physics_actuator_torque',physics_warning_number='physics_warning_counts',
        physics_warning_lastinfo='physics_warning_lastinfo')
    mapping.update({'control_history_'+key:'control_history_before_'+key for key in KEYS})
    for combined, baseline_key in mapping.items():
        np.testing.assert_array_equal(trace[combined][:len(baseline[baseline_key])],baseline[baseline_key])
    np.testing.assert_array_equal(trace['initial_integration'],baseline['initial_integration'])
    np.testing.assert_array_equal(trace['branch_initial_integration'],baseline['final_integration'])
    np.testing.assert_array_equal(trace['control_integration_before'][250],snapshot['integration'])
    np.testing.assert_array_equal(trace['control_previous_action_before'][250],snapshot['previous_action'])
    np.testing.assert_array_equal(trace['control_history_before'][250],snapshot['history_flat'])
    for key in KEYS:np.testing.assert_array_equal(trace['control_history_'+key][250],snapshot['history_'+key])
    accumulated=0.
    assert trace['physics_expected_time'][0]==0.
    for i in range(15690):
        accumulated+=.002
        assert accumulated==trace['physics_expected_time'][i+1]==trace['physics_time'][i+1]
    limits = np.asarray(c['joint_limits'])
    seed = Native23BFMRolloutSeed(native, c, original, ONNX, dependency_directory=DEPS, threads=1)
    feature_builder = LinearFeatures(motion, original29, c)
    history = BFMHistory()
    previous = np.zeros(23, np.float32)
    names = ('features', 'residual_rad', 'base_target', 'expert_target', 'base_action',
             'previous_action', 'history', 'state', 'source_frame', 'control')
    rows = {key: [] for key in names}
    measured_names = {'history_' + key: [] for key in KEYS}
    started = time.perf_counter()
    for control in range(1269):
        qpos, qvel = trace['qpos'][control], trace['qvel'][control]
        np.testing.assert_array_equal(previous, trace['control_previous_action_before'][control])
        for key in KEYS:
            np.testing.assert_array_equal(history.data[key], trace['control_history_' + key][control])
        sensed, terms = seed._terms(qpos, qvel, previous)
        saved_names = {key: history.data[key].copy() for key in KEYS}
        flat_history = history.before_update(terms)
        np.testing.assert_array_equal(flat_history, trace['control_history_before'][control])
        target = trace['target'][control]
        if control >= 250:
            raw, base, _ = infer_base(seed, qpos, qvel, previous, flat_history, control + 11)
            features = feature_builder(qpos, qvel, control + 11, base, previous)
            assert features.shape == (FEATURES,)
            assert np.isfinite(target).all()
            np.testing.assert_array_equal(target, np.clip(target, limits[:, 0], limits[:, 1]))
            values = dict(features=features, residual_rad=target-base, base_target=base,
                expert_target=target, base_action=raw, previous_action=previous.copy(),
                history=flat_history, state=sensed, source_frame=control+11, control=control)
            for key, value in values.items():
                rows[key].append(value)
            for key in KEYS:
                measured_names['history_' + key].append(saved_names[key])
        # Original BFM controls0..249 are provenance only; preserve their raw actions.
        previous = (trace['action'][control].copy() if control < 250 else
            ((target - np.asarray(c['default_q'])) * np.asarray(c['kp']) /
             (.25 * np.asarray(c['training_effort']))).astype(np.float32))
        np.testing.assert_array_equal(previous, trace['action'][control])
    arrays = {key: np.asarray(values) for key, values in rows.items()}
    np.testing.assert_array_equal(arrays['control'], np.arange(250, 1269))
    assert all(np.isfinite(value).all() for value in arrays.values())
    np.testing.assert_array_equal(arrays['source_frame'],np.arange(261,1280))
    np.testing.assert_array_equal(previous,trace['control_previous_action_before'][1269])
    for key in KEYS:np.testing.assert_array_equal(history.data[key],trace['control_history_'+key][1269])
    query_parity={}
    for key, actual, expected in (
        ('previous_action',arrays['previous_action'][0],snapshot['previous_action']),
        ('history',arrays['history'][0],snapshot['history_flat']),
        ('qpos',trace['qpos'][250],snapshot['qpos']),('qvel',trace['qvel'][250],snapshot['qvel']),
        ('integration',trace['control_integration_before'][250],snapshot['integration'])):
        np.testing.assert_array_equal(actual,expected)
        query_parity[key]=dict(bitexact=True,max_abs_difference=0.)
    first_fresh=archive(RUN/'initial_seed/fresh_bfm.npz')['targets'][0]
    np.testing.assert_array_equal(np.clip(arrays['base_target'][0],limits[:,0],limits[:,1]),first_fresh)
    query_parity['clipped_base_vs_initial_fresh_seed']=dict(bitexact=True,max_abs_difference=0.)
    span = np.diff(limits, axis=1).ravel().astype(np.float32)
    old = archive(PILOT / 'labels/labels.npz')
    np.testing.assert_array_equal(span, old['joint_span'])
    np.testing.assert_array_equal(limits, old['joint_limits'])
    np.savez_compressed(destination / 'labels.npz', **arrays, joint_span=span, joint_limits=limits,
        teacher_qpos=trace['qpos'][250:1270], teacher_qvel=trace['qvel'][250:1270],
        control_integration_before=trace['control_integration_before'][250:1269],
        **{key: np.asarray(value) for key, value in measured_names.items()})
    normalization = archive(STUDENT / 'fit/teacher_fit.npz')
    original_normalization=archive(TASK/'fast_controller_continued_fit_v1/fit/teacher_fit.npz')
    for key in ('feature_mean','feature_std'):np.testing.assert_array_equal(normalization[key],original_normalization[key])
    np.savez_compressed(destination / 'existing_normalization.npz',
        feature_mean=normalization['feature_mean'], feature_std=normalization['feature_std'])
    result = dict(kind='one_qualified_BFM250_actual_expert_transition_labels', samples=1019,
        expert_controls=[250, 1268], excluded_BFM_prefix_controls=[0,249], excluded_terminal_controls=[1269, 1818],
        native_applied_target_labels_only=True, named_history_and_previous_action_exact=True,
        query250_input_and_initial_base_parity=query_parity, labels_sha256=sha(destination / 'labels.npz'),
        reconstruction_max_abs_rad=float(np.max(np.abs(arrays['base_target'] + arrays['residual_rad'] - arrays['expert_target']))),
        previous_action_max_abs=float(np.max(np.abs(arrays['previous_action']))),
        previous_action_components_outside_five=int(np.sum(np.abs(arrays['previous_action']) > 5)),
        features=FEATURES, goal_offsets=OFFSETS.tolist(), clock_or_clip_id_features=False,
        received_goal_preview_seconds=.74, raw_pose_support_seconds=.76,
        frozen_base_goal='original native h8/position1/yaw2', seed_identity=seed.identity(),
        qualification_receipt_sha256=sha(args.qualification), qualification=decision,
        collector_frozen_receipt_sha256=sha(BASE / 'collector_frozen_inputs.json'),
        expert_branch_receipt_sha256=frozen['expert_branch_receipt_sha256'],
        expert_launch_clearance_sha256=frozen['expert_launch_clearance_sha256'],
        baseline250_sha256=frozen['baseline250_sha256'],
        normalization_refitted=False, optimizer_calls=0, physics_steps=0, fitting_launched=False,
        actor_inference_calls=1019, elapsed_seconds=time.perf_counter()-started, hardware_authorized=False)
    write(destination / 'report.json', result)
    print(json.dumps(dict(samples=1019, labels_sha256=result['labels_sha256'])), flush=True)


if __name__ == '__main__':
    main()
