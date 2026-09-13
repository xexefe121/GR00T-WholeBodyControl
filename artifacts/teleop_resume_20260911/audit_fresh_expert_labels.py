"""Independent saved-array checks of actual expert labels and neighbor witnesses."""

import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with np.load(path, allow_pickle=False) as a:
        return {k: a[k].copy() for k in a.files}


def main():
    folder = BASE / 'fresh_expert_labels_resume_v1'
    paths = dict(new=folder / 'labels/labels.npz',
                 old=BASE / 'fast_controller_nominal_pilot_v1/labels/labels.npz',
                 trace=BASE / 'student_actual_oracle_control1_resume1001_v1/nominal/trace.npz',
                 norm=folder / 'labels/existing_normalization.npz',
                 fit=BASE / 'fast_controller_continued_fit_v1/fit/teacher_fit.npz',
                 pairs=folder / 'compatibility/cross_distances_and_pairs.npz')
    new, old, trace, norm, fit, pairs = [read(p) for p in paths.values()]
    controls = np.arange(1, 1269)
    np.testing.assert_array_equal(new['control'], controls)
    np.testing.assert_array_equal(new['expert_target'], trace['target'][controls])
    np.testing.assert_array_equal(new['source_frame'], trace['source_frame'][controls])
    np.testing.assert_array_equal(new['control_integration_before'], trace['control_integration_before'][controls])
    np.testing.assert_array_equal(new['history'], trace['control_history_before'][controls])
    np.testing.assert_array_equal(new['previous_action'], trace['control_previous_action_before'][controls])
    for key in ('actions', 'base_ang_vel', 'dof_pos', 'dof_vel', 'projected_gravity'):
        np.testing.assert_array_equal(new['history_' + key], trace['control_history_' + key][controls])
    np.testing.assert_array_equal(new['teacher_qpos'], trace['qpos'][1:1270])
    np.testing.assert_array_equal(new['teacher_qvel'], trace['qvel'][1:1270])
    np.testing.assert_array_equal(new['joint_limits'], old['joint_limits'])
    np.testing.assert_array_equal(new['joint_span'], old['joint_span'])
    for key in ('feature_mean', 'feature_std'):
        np.testing.assert_array_equal(norm[key], fit[key])
    assert np.all(new['expert_target'] >= new['joint_limits'][:, 0])
    assert np.all(new['expert_target'] <= new['joint_limits'][:, 1])
    np.testing.assert_allclose(new['base_target'] + new['residual_rad'], new['expert_target'], atol=3e-16, rtol=0)
    distance = pairs['normalized_feature_rms_distance_float64']
    nearest = np.argmin(distance, axis=1)
    np.testing.assert_array_equal(nearest, pairs['nearest_old_row'])
    far = distance.copy()
    far[np.abs(controls[:, None] - old['control'][None, :]) <= 4] = np.inf
    np.testing.assert_array_equal(np.argmin(far, axis=1), pairs['nearest_outside_plusminus4_old_row'])
    for name in ('nearest', 'nearest_outside_plusminus4'):
        ids = pairs[name + '_old_row']
        np.testing.assert_array_equal(pairs[name + '_target_delta'], new['expert_target'] - old['expert_target'][ids])
        np.testing.assert_array_equal(pairs[name + '_residual_delta'], new['residual_rad'] - old['residual_rad'][ids])
    # Check deterministic spread of rows plus every close or high-sensitivity witness.
    selected = np.unique(np.r_[np.linspace(0, 1267, 24).astype(int),
                               np.argsort(pairs['nearest_feature_rms'])[:20],
                               np.argsort(pairs['nearest_sensitivity_ratio'])[-20:]])
    mean, std = norm['feature_mean'].astype(float), norm['feature_std'].astype(float)
    y = (old['features'].astype(float) - mean) / std
    max_error = 0.
    for i in selected:
        x = (new['features'][i].astype(float) - mean) / std
        value = np.sqrt(np.sum((x - y) ** 2, axis=1) / 1069)
        max_error = max(max_error, float(np.max(np.abs(value - distance[i]))))
        np.testing.assert_allclose(value, distance[i], rtol=2e-14, atol=2e-15)
    all_features = np.r_[old['features'], new['features']].astype(np.float32)
    all_features[all_features == 0] = 0.
    new_duplicates = sum(any(np.array_equal(row, earlier) for earlier in all_features[:1269 + i])
                         for i, row in enumerate(new['features']))
    assert new_duplicates == 0
    out = BASE / 'fresh_expert_labels_independent_v1'
    out.mkdir(exist_ok=False)
    report = dict(independent_saved_array_label_checks_pass=True,
                  all_actual_targets_and_measured_histories_bitexact=True,
                  old_normalization_unchanged=True, admissible_expert_controls=[1, 1268],
                  student_control0_excluded=True, terminal_BFM_excluded=True,
                  recomputed_cross_distance_rows=selected.tolist(),
                  recomputed_cross_distance_pairs=len(selected) * 1269,
                  maximum_cross_distance_absolute_difference=max_error,
                  exact_duplicate_new_rows=new_duplicates,
                  labels_smoothed_removed_or_averaged=False,
                  physics_steps=0, inference_calls=0, optimizer_calls=0,
                  hashes={str(p): sha(p) for p in [*paths.values(), Path(__file__)]})
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    (out / 'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k: v for k, v in report.items() if k not in ('hashes', 'recomputed_cross_distance_rows')}))


if __name__ == '__main__':
    main()
