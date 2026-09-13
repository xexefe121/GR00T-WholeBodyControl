"""Saved-array label compatibility audit; no inference, training or physics."""
import argparse
from pathlib import Path
import numpy as np
from collect_actual_branch_labels import BASE, PILOT, STUDENT, archive, read, write, sha, local


def distribution(value):
    value = np.asarray(value, np.float64)
    return dict(minimum=float(np.min(value)), median=float(np.median(value)),
                p95=float(np.percentile(value, 95)), maximum=float(np.max(value)))


def phase(control):
    return np.where(control < 250, 0, np.where(control < 350, 1, np.where(control < 1169, 2, 3)))


def duplicate_groups(old, new):
    buckets = {}
    combined_features = np.concatenate((old['features'], new['features'])).astype(np.float32)
    targets = np.concatenate((old['expert_target'], new['expert_target']))
    residuals = np.concatenate((old['residual_rad'], new['residual_rad']))
    # Numeric equality treats signed zero equally; canonicalize it before hashing.
    combined_features[combined_features == 0] = 0.
    for index, row in enumerate(combined_features):
        buckets.setdefault(row.tobytes(), []).append(index)
    groups = []
    for indices in buckets.values():
        if len(indices) < 2 or max(indices) < len(old['features']):
            continue
        target_spread = np.ptp(targets[indices], axis=0)
        residual_spread = np.ptp(residuals[indices], axis=0)
        groups.append(dict(rows=[dict(dataset='old' if i < 1269 else 'new',
            row=i if i < 1269 else i-1269,
            control=int(old['control'][i] if i < 1269 else new['control'][i-1269])) for i in indices],
            expert_target_spread_by_joint_rad=target_spread.tolist(),
            residual_spread_by_joint_rad=residual_spread.tolist(),
            deterministic_target_conflict=bool(np.any(target_spread != 0)),
            deterministic_residual_conflict=bool(np.any(residual_spread != 0))))
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels-report', type=Path, default=BASE / 'labels/report.json')
    args = parser.parse_args()
    frozen = read(BASE / 'collector_frozen_inputs.json')
    for name, digest in frozen['source_sha256'].items():
        assert sha(Path(__file__).parent / name) == digest, name
    for name, digest in frozen['input_sha256'].items():
        assert sha(local(name)) == digest, name
    joint_names = read(local(frozen['contract_path']))['joint_names']
    receipt = read(args.labels_report)
    assert receipt['samples'] == 1268 and receipt['fitting_launched'] is False
    assert receipt['collector_frozen_receipt_sha256'] == sha(BASE / 'collector_frozen_inputs.json')
    new_path = args.labels_report.parent / 'labels.npz'
    assert sha(new_path) == receipt['labels_sha256']
    old_path = PILOT / 'labels/labels.npz'
    assert sha(old_path) == '8003407282b01f2c66fe1720d7eb038f43377c1d6eab80dce47f1f75911e8083'
    new, old = archive(new_path), archive(old_path)
    np.testing.assert_array_equal(new['control'], np.arange(1, 1269))
    np.testing.assert_array_equal(new['source_frame'], np.arange(12, 1280))
    normalization = archive(args.labels_report.parent / 'existing_normalization.npz')
    original_normalization = archive(STUDENT / 'fit/teacher_fit.npz')
    for key in ('feature_mean', 'feature_std'):
        np.testing.assert_array_equal(normalization[key], original_normalization[key])
    mean, std = [normalization[key].astype(np.float64) for key in ('feature_mean', 'feature_std')]
    assert np.all(std >= .05 - 1e-9)
    X = (new['features'].astype(np.float64) - mean) / std
    Y = (old['features'].astype(np.float64) - mean) / std
    assert X.shape == (1268, 1069) and Y.shape == (1269, 1069)
    # Explicit float64 differences avoid cancellation in norm-sum dot products.
    distances = np.empty((1268, 1269), np.float64)
    for start in range(0, 1268, 8):
        delta = X[start:start+8, None] - Y[None]
        distances[start:start+8] = np.sqrt(np.einsum('ijk,ijk->ij', delta, delta,
                                                     optimize=False) / 1069)
    closest = np.argmin(distances, axis=1)
    outside = distances.copy()
    outside[np.abs(new['control'][:, None] - old['control'][None]) <= 4] = np.inf
    nonlocal_closest = np.argmin(outside, axis=1)
    selected = {}
    pair_arrays = {}
    row_indices = np.arange(1268)
    for name, indices in [('nearest', closest), ('nearest_outside_plusminus4', nonlocal_closest)]:
        distance = distances[row_indices, indices]
        target_delta = new['expert_target'] - old['expert_target'][indices]
        residual_delta = new['residual_rad'] - old['residual_rad'][indices]
        target_rms = np.sqrt(np.mean(target_delta**2, axis=1))
        residual_rms = np.sqrt(np.mean(residual_delta**2, axis=1))
        ratio = np.full(1268, np.nan)
        np.divide(target_rms, distance, out=ratio, where=distance > 0)
        selected[name] = dict(normalized_feature_rms_distance=distribution(distance),
            target_rms_difference_rad=distribution(target_rms), residual_rms_difference_rad=distribution(residual_rms),
            exact_zero_distance_count=int(np.sum(distance == 0)),
            zero_distance_target_conflict_count=int(np.sum((distance == 0) & (target_rms != 0))),
            twenty_closest_new_row_indices=np.argsort(distance)[:20].tolist(),
            twenty_largest_finite_sensitivity_new_row_indices=np.argsort(np.where(np.isfinite(ratio), ratio, -1))[-20:][::-1].tolist(),
            largest_difference_joint_index=int(np.unravel_index(np.argmax(np.abs(target_delta)), target_delta.shape)[1]))
        selected[name]['largest_difference_joint_name'] = joint_names[selected[name]['largest_difference_joint_index']]
        pair_arrays.update({name + '_' + key: value for key, value in dict(old_row=indices,
            new_control=new['control'], old_control=old['control'][indices], feature_rms=distance,
            target_delta=target_delta, residual_delta=residual_delta, target_rms=target_rms,
            residual_rms=residual_rms, sensitivity_ratio=ratio).items()})
    same = new['control'].astype(np.int64)
    np.testing.assert_array_equal(old['control'][same], new['control'])
    same_arrays = {}
    for key in ('features', 'history', 'previous_action', 'base_target', 'expert_target', 'residual_rad'):
        same_arrays[key] = new[key].astype(np.float64) - old[key][same].astype(np.float64)
    same_arrays['qpos'] = new['teacher_qpos'][:-1] - old['teacher_qpos'][same]
    same_arrays['qvel'] = new['teacher_qvel'][:-1] - old['teacher_qvel'][same]
    same_arrays['proprio_features'] = same_arrays['features'][:, :79]
    same_arrays['goal_features'] = same_arrays['features'][:, 79:1023]
    groups = duplicate_groups(old, new)
    strata = {}
    for index, name in enumerate(('entry', 'acquisition', 'source', 'return')):
        chosen = phase(new['control']) == index
        strata[name] = dict(rows=int(np.sum(chosen)), **{
            key + '_rms_difference': distribution(np.sqrt(np.mean(values[chosen]**2, axis=1)))
            for key, values in same_arrays.items()})
    command = {}
    for name, data in [('old', old), ('new', new)]:
        slew = np.diff(data['expert_target'], axis=0)
        span = data['joint_span'].astype(np.float64)
        command[name] = dict(target_slew_joint_l2_rad_per_20ms=distribution(np.linalg.norm(slew, axis=1)),
            target_slew_abs_by_joint_p95_rad=np.percentile(np.abs(slew), 95, axis=0).tolist(),
            residual_abs_by_joint_p95_rad=np.percentile(np.abs(data['residual_rad']), 95, axis=0).tolist(),
            residual_abs_native_span_by_joint_p95=np.percentile(np.abs(data['residual_rad'])/span, 95, axis=0).tolist(),
            previous_action_max_abs=float(np.max(np.abs(data['previous_action']))),
            previous_action_components_outside_five=int(np.sum(np.abs(data['previous_action']) > 5)))
    destination = BASE / 'compatibility'
    destination.mkdir(exist_ok=False)
    np.savez_compressed(destination / 'cross_distances_and_pairs.npz',
        normalized_feature_rms_distance_float64=distances, **pair_arrays,
        **{'same_control_' + key: value for key, value in same_arrays.items()})
    write(destination / 'exact_duplicate_groups.json', groups)
    write(destination / 'report.json', dict(kind='saved_array_cross_dataset_compatibility',
        old_rows=1269, new_rows=1268, original_normalization_refitted=False,
        native_joint_names=joint_names,
        distance_arithmetic='explicit float64 normalized differences and squared sum; RMS over1069 features',
        exact_duplicate_groups=len(groups), deterministic_conflict_groups=sum(g['deterministic_target_conflict'] for g in groups),
        deterministic_target_conflict_groups=sum(g['deterministic_target_conflict'] for g in groups),
        deterministic_residual_conflict_groups=sum(g['deterministic_residual_conflict'] for g in groups),
        nearest_neighbor=selected, same_control_by_phase=strata, command_statistics=command,
        saved_pairs_sha256=sha(destination / 'cross_distances_and_pairs.npz'),
        old_labels_sha256=sha(old_path), new_labels_sha256=sha(new_path),
        labels_report_sha256=sha(args.labels_report), normalization_source_sha256=sha(STUDENT / 'fit/teacher_fit.npz'),
        target_smoothing_or_removal=False, labels_averaged=False, fitting_launched=False,
        inference_calls=0, optimizer_calls=0, physics_steps=0, hardware_authorized=False))
    print('Saved-array compatibility audit complete; no fitting launched.', flush=True)


if __name__ == '__main__':
    main()
