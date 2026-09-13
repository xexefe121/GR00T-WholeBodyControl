"""Saved-only first15 aggregate student diagnosis: no inference or dynamics."""
from pathlib import Path
import hashlib
import json
import numpy as np

BASE = Path(__file__).resolve().parent
TASK = BASE.parent
RUN = TASK / 'fast_controller_aggregate_fit_v1'


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rms(value, axis=None):
    return np.sqrt(np.mean(np.asarray(value, np.float64)**2, axis=axis))


def main():
    paths = dict(actual=RUN/'nominal/trace.npz', old=TASK/'fast_controller_nominal_pilot_v1/labels/labels.npz',
        new=TASK/'fresh_expert_labels_resume_v1/labels/labels.npz', fit=RUN/'fit/teacher_fit.npz',
        fit_metrics=RUN/'fit/full_fit_metrics.json',
        contract=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json'))
    actual, old, new, fit = [load(paths[key]) for key in ('actual', 'old', 'new', 'fit')]
    contract = json.loads(paths['contract'].read_text())
    names = contract['joint_names']
    limits = np.asarray(contract['joint_limits'])
    count = len(actual['target'])
    assert count == 15 and len(actual['physics_torque']) == 147
    np.testing.assert_array_equal(actual['global_control'], np.arange(count))
    np.testing.assert_array_equal(actual['previous_action'][1:], actual['action'][:-1])
    features = np.concatenate((old['features'], new['features'])).astype(np.float64)
    target = np.concatenate((old['expert_target'], new['expert_target']))
    controls = np.concatenate((old['control'], new['control']))
    dataset = np.r_[np.zeros(1269, np.int64), np.ones(1268, np.int64)]
    mean, std = fit['feature_mean'].astype(np.float64), fit['feature_std'].astype(np.float64)
    X = (actual['features'].astype(np.float64)-mean)/std
    Y = (features-mean)/std
    distances = np.empty((count, 2537), np.float64)
    for start in range(0, count, 3):
        delta = X[start:start+3, None]-Y[None]
        distances[start:start+3] = rms(delta, axis=2)
    closest = np.argmin(distances, axis=1)
    low, high = features.min(0), features.max(0)
    outside = np.maximum(np.maximum(low-actual['features'], actual['features']-high), 0)/std
    raw_targets = actual['base_target']+actual['delta']
    clip_delta = raw_targets-actual['target']
    prior_min, prior_max = features[:, -23:].min(0), features[:, -23:].max(0)
    prior_outside = (actual['previous_action'] < prior_min) | (actual['previous_action'] > prior_max)
    block_slices = dict(proprio=slice(0, 79), future_goal=slice(79, 1023),
                        BFM_base_offset=slice(1023, 1046), previous_action=slice(1046, 1069))
    exact0 = {}
    for key in ('features', 'base_target', 'previous_action', 'history', 'state'):
        exact0[key] = bool(np.array_equal(actual[key][0], old[key][0]))
    exact0['qpos'] = bool(np.array_equal(actual['qpos'][0], old['teacher_qpos'][0]))
    exact0['qvel'] = bool(np.array_equal(actual['qvel'][0], old['teacher_qvel'][0]))
    assert all(exact0.values())
    rows = []
    teacher_errors = {}
    for label, data, offset in [('old', old, 0), ('new', new, 1269)]:
        start = 0 if label == 'old' else 1
        ids = np.arange(start, count)
        indices = ids if label == 'old' else ids-1
        teacher_errors[label] = dict(global_control=ids,
            actual_target_minus_expert=actual['target'][ids]-data['expert_target'][indices],
            saved_fit_target_minus_expert=fit['predicted_applied_target'][offset+indices]-data['expert_target'][indices],
            actual_features_minus_teacher=actual['features'][ids]-data['features'][indices],
            normalized_feature_rms=rms((actual['features'][ids]-data['features'][indices])/std, axis=1),
            qpos_difference=actual['qpos'][ids]-data['teacher_qpos'][indices],
            qvel_difference=actual['qvel'][ids]-data['teacher_qvel'][indices])
    for control in range(count):
        nearest = int(closest[control])
        raw = raw_targets[control]
        mismatch = actual['target'][control]-old['expert_target'][control]
        worst = int(np.argmax(np.abs(mismatch)))
        row = dict(control=control, executed_substeps=int(actual['physics_substeps'][control]),
            nearest=dict(dataset='old' if dataset[nearest] == 0 else 'new', row=nearest if nearest < 1269 else nearest-1269,
                control=int(controls[nearest]), normalized_feature_rms=float(distances[control, nearest]),
                actual_target_error_rms_rad=float(rms(actual['target'][control]-target[nearest])),
                saved_fit_target_error_rms_rad=float(rms(fit['predicted_applied_target'][nearest]-target[nearest])),
                actual_vs_saved_fit_prediction_rms_rad=float(rms(actual['target'][control]-fit['predicted_applied_target'][nearest]))),
            same_control_old=dict(actual_target_error_rms_rad=float(rms(mismatch)),
                saved_fit_target_error_rms_rad=float(rms(teacher_errors['old']['saved_fit_target_minus_expert'][control])),
                standardized_feature_rms=float(teacher_errors['old']['normalized_feature_rms'][control]),
                precontrol_max_joint_position_difference_rad=float(np.max(np.abs(teacher_errors['old']['qpos_difference'][control, 7:]))),
                precontrol_max_joint_velocity_difference_radps=float(np.max(np.abs(teacher_errors['old']['qvel_difference'][control, 6:]))),
                worst_target_joint=names[worst], worst_target_difference_rad=float(mismatch[worst])),
            standardized_feature_block_differences_to_nearest={key: float(rms(X[control, sl]-Y[nearest, sl])) for key, sl in block_slices.items()},
            features_outside_combined_training_minmax_count=int(np.sum(outside[control] > 0)),
            feature_max_standardized_excursion_outside_training=float(np.max(outside[control])),
            previous_action_outside_training_joint_ranges=[names[j] for j in np.flatnonzero(prior_outside[control])],
            previous_action_max_abs=float(np.max(np.abs(actual['previous_action'][control]))),
            produced_combined_action_max_abs=float(np.max(np.abs(actual['action'][control]))),
            clipped_target_joints=[names[j] for j in np.flatnonzero(np.abs(clip_delta[control]) > 1e-12)],
            target_clip_max_abs_rad=float(np.max(np.abs(clip_delta[control]))),
            learned_correction_max_abs_rad=float(np.max(np.abs(actual['delta'][control]))),
            BFM_base_max_abs_rad=float(np.max(np.abs(actual['base_target'][control]))),
            right_knee=dict(base=float(actual['base_target'][control, 9]),
                delta=float(actual['delta'][control, 9]), raw_target=float(raw[9]),
                applied_target=float(actual['target'][control, 9]),
                actual_precontrol_position=float(actual['qpos'][control, 16]),
                actual_precontrol_velocity=float(actual['qvel'][control, 15])))
        if control:
            row['same_control_new'] = dict(actual_target_error_rms_rad=float(rms(teacher_errors['new']['actual_target_minus_expert'][control-1])),
                saved_fit_target_error_rms_rad=float(rms(teacher_errors['new']['saved_fit_target_minus_expert'][control-1])),
                standardized_feature_rms=float(teacher_errors['new']['normalized_feature_rms'][control-1]))
        rows.append(row)
    metrics = json.loads(paths['fit_metrics'].read_text())
    last = metrics[-11:]
    trend = [dict(step=m['global_step'], old_rmse=m['datasets']['old']['all']['applied_target_rmse_rad'],
        new_rmse=m['datasets']['new']['all']['applied_target_rmse_rad'],
        query1_rmse=m['actual_student_query_control1']['target_rmse_rad']) for m in last]
    first_clip = np.flatnonzero(np.any(np.abs(clip_delta) > 1e-12, axis=1))
    first_prior_outside = np.flatnonzero(np.any(prior_outside, axis=1))
    metadata = dict(kind='saved_only_aggregate_canonical_failure_diagnosis', controls=count, physics_steps=147,
        control0_exact_old_teacher_input=exact0,
        control0_ONNX_actual_vs_saved_Torch_target_max_difference_rad=float(np.max(np.abs(actual['target'][0]-fit['predicted_applied_target'][0]))),
        control0_target_error_rms_rad=float(rms(actual['target'][0]-old['expert_target'][0])),
        first_native_target_clip_control=int(first_clip[0]) if len(first_clip) else None,
        first_previous_action_outside_any_training_joint_range=int(first_prior_outside[0]) if len(first_prior_outside) else None,
        observed_previous_action_max_abs=float(np.max(np.abs(actual['previous_action']))),
        observed_produced_action_max_abs=float(np.max(np.abs(actual['action']))),
        training_previous_action_max_abs=float(np.max(np.abs(features[:, -23:]))),
        last10000_fit_diagnostics=trend, actual_rows=rows,
        input_sha256={key: sha(path) for key, path in paths.items()},
        script_sha256=sha(Path(__file__)), inference_calls=0, optimizer_calls=0, new_physics_steps=0,
        causal_intervention_performed=False, insufficient_observation_proved=False, hardware_authorized=False)
    (BASE/'report.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    np.savez_compressed(BASE/'arrays.npz', standardized_cross_feature_distances=distances,
        nearest_indices=closest, standardized_feature_excursion=outside, native_target_clip_delta=clip_delta,
        actual_raw_target=raw_targets, previous_action_outside_training_joint_ranges=prior_outside,
        **{label+'_'+key: value for label, data in teacher_errors.items() for key, value in data.items()})
    print(json.dumps({key: metadata[key] for key in ('control0_target_error_rms_rad',
        'first_native_target_clip_control', 'first_previous_action_outside_any_training_joint_range',
        'observed_previous_action_max_abs', 'training_previous_action_max_abs')}), flush=True)


if __name__ == '__main__':
    main()
