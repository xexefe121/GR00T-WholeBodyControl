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
RUN = TASK / 'student_actual_oracle_control1_resume1001_v1'
ORIGINAL = TASK / 'student_actual_oracle_control1_v1'
PILOT = TASK / 'fast_controller_nominal_pilot_v1'
STUDENT = TASK / 'fast_controller_continued_fit_v1'
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
    assert sha(RUN / 'resume_inputs/trace.partial.npz') == frozen['checkpoint1001_sha256']
    assert sha(ORIGINAL / 'frozen_inputs_v2.json') == frozen['original_v2_receipt_sha256']
    assert sha(RUN / 'resume_receipt.json') == frozen['resume_receipt_sha256']
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
    checkpoint = archive(RUN / 'resume_inputs/trace.partial.npz')
    extension = archive(RUN / 'post_lifecycle_hold_5s/trace.npz')
    np.testing.assert_array_equal(trace['global_control'], np.arange(1569))
    np.testing.assert_array_equal(trace['source_frame'], np.arange(1569) + 11)
    np.testing.assert_array_equal(trace['physics_substeps'], np.full(1569, 10))
    np.testing.assert_array_equal(trace['controller_mode'], np.r_[0, np.ones(1268), np.full(300, 2)])
    np.testing.assert_array_equal(extension['global_control'], np.arange(1569, 1819))
    np.testing.assert_array_equal(extension['physics_substeps'], np.full(250, 10))
    np.testing.assert_array_equal(trace['final_integration'], extension['initial_integration'])
    np.testing.assert_array_equal(trace['final_previous_action'], extension['control_previous_action_before'][0])
    for key in KEYS:
        np.testing.assert_array_equal(trace['final_history_' + key], extension['control_history_' + key][0])
    for key, old in checkpoint.items():
        if key.startswith('final_') or key == 'checkpoint_metadata':
            continue
        actual = trace[key] if key in ('initial_integration', 'branch_initial_integration', 'integration_state_spec') else trace[key][:len(old)]
        np.testing.assert_array_equal(old, actual)
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
        if control > 0:
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
        # Student control0 is provenance only and keeps its combined PRECLIP action.
        previous = (trace['action'][0].copy() if control == 0 else
            ((target - np.asarray(c['default_q'])) * np.asarray(c['kp']) /
             (.25 * np.asarray(c['training_effort']))).astype(np.float32))
        np.testing.assert_array_equal(previous, trace['action'][control])
    arrays = {key: np.asarray(values) for key, values in rows.items()}
    np.testing.assert_array_equal(arrays['control'], np.arange(1, 1269))
    assert all(np.isfinite(value).all() for value in arrays.values())
    student = archive(STUDENT / 'nominal/trace.npz')
    query_parity = {}
    for key in ('features', 'base_target', 'previous_action', 'history', 'state'):
        error = float(np.max(np.abs(arrays[key][0] - student[key][1])))
        query_parity[key] = dict(max_abs_difference=error, bitexact=bool(np.array_equal(arrays[key][0], student[key][1])))
        assert error <= (1e-5 if key in ('features', 'base_target') else 0.), (key, error)
    span = np.diff(limits, axis=1).ravel().astype(np.float32)
    old = archive(PILOT / 'labels/labels.npz')
    np.testing.assert_array_equal(span, old['joint_span'])
    np.testing.assert_array_equal(limits, old['joint_limits'])
    np.savez_compressed(destination / 'labels.npz', **arrays, joint_span=span, joint_limits=limits,
        teacher_qpos=trace['qpos'][1:1270], teacher_qvel=trace['qvel'][1:1270],
        control_integration_before=trace['control_integration_before'][1:1269],
        **{key: np.asarray(value) for key, value in measured_names.items()})
    normalization = archive(STUDENT / 'fit/teacher_fit.npz')
    np.savez_compressed(destination / 'existing_normalization.npz',
        feature_mean=normalization['feature_mean'], feature_std=normalization['feature_std'])
    result = dict(kind='one_qualified_actual_student_state_expert_branch_labels', samples=1268,
        expert_controls=[1, 1268], excluded_student_prefix_control=0, excluded_terminal_controls=[1269, 1818],
        native_applied_target_labels_only=True, named_history_and_previous_action_exact=True,
        query_input_student_parity=query_parity, labels_sha256=sha(destination / 'labels.npz'),
        reconstruction_max_abs_rad=float(np.max(np.abs(arrays['base_target'] + arrays['residual_rad'] - arrays['expert_target']))),
        previous_action_max_abs=float(np.max(np.abs(arrays['previous_action']))),
        previous_action_components_outside_five=int(np.sum(np.abs(arrays['previous_action']) > 5)),
        features=FEATURES, goal_offsets=OFFSETS.tolist(), clock_or_clip_id_features=False,
        received_goal_preview_seconds=.74, raw_pose_support_seconds=.76,
        frozen_base_goal='original native h8/position1/yaw2', seed_identity=seed.identity(),
        qualification_receipt_sha256=sha(args.qualification), qualification=decision,
        collector_frozen_receipt_sha256=sha(BASE / 'collector_frozen_inputs.json'),
        original_v2_receipt_sha256=frozen['original_v2_receipt_sha256'],
        resume_receipt_sha256=frozen['resume_receipt_sha256'],
        checkpoint1001_sha256=frozen['checkpoint1001_sha256'],
        normalization_refitted=False, optimizer_calls=0, physics_steps=0, fitting_launched=False,
        actor_inference_calls=1268, elapsed_seconds=time.perf_counter()-started, hardware_authorized=False)
    write(destination / 'report.json', result)
    print(json.dumps(dict(samples=1268, labels_sha256=result['labels_sha256'])), flush=True)


if __name__ == '__main__':
    main()
