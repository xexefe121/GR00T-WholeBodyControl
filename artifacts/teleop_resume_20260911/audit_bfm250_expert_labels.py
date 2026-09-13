"""Independent saved-array verification of the moving-phase expert dataset."""

import json
from pathlib import Path

import numpy as np

from .inspect_recorded_intent import ROOT, sha
from gear_sonic.utils.g1_true23_mpc_student import GoalFeatures

BASE = Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')


def read(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def main():
    folder = BASE / 'bfm_entry250_labels_v1'
    paths = dict(new=folder / 'labels/labels.npz',
        old=BASE / 'fast_controller_nominal_pilot_v1/labels/labels.npz',
        query1=BASE / 'fresh_expert_labels_resume_v1/labels/labels.npz',
        trace=BASE / 'bfm_entry250_actual_oracle_v1/nominal/trace.npz',
        norm=folder / 'labels/existing_normalization.npz',
        fit=BASE / 'fast_controller_aggregate_fit_v1/fit/teacher_fit.npz',
        pairs=folder / 'compatibility/distances_and_pairs.npz')
    new, old, query1, trace, norm, fit, pairs = [read(p) for p in paths.values()]
    qualification_path = BASE / 'bfm250_expert_root_qualification_v1/qualification.json'
    qualification = json.loads(qualification_path.read_text())
    assert qualification['root_authorized_extraction'] and not qualification['model_fitting_authorized']
    assert sha(paths['trace']) == qualification['nominal_trace_sha256']
    label_report_path = folder / 'labels/report.json'
    label_report = json.loads(label_report_path.read_text())
    assert label_report['qualification_receipt_sha256'] == sha(qualification_path)
    assert label_report['labels_sha256'] == sha(paths['new']) and label_report['samples'] == 1019
    compatibility_path = folder / 'compatibility/report.json'
    compatibility = json.loads(compatibility_path.read_text())
    assert compatibility['compatibility_pass'] and compatibility['exact_conflict_groups'] == 0
    assert compatibility['distance_matrices_sha256'] == sha(paths['pairs'])
    for name in ('old', 'query1'):
        assert compatibility['label_sha256'][name] == sha(paths[name])
    assert compatibility['label_sha256']['query250'] == sha(paths['new'])
    controls = np.arange(250, 1269)
    np.testing.assert_array_equal(new['control'], controls)
    for label, recorded in (('expert_target', 'target'), ('source_frame', 'source_frame'),
        ('control_integration_before', 'control_integration_before'),
        ('history', 'control_history_before'), ('previous_action', 'control_previous_action_before')):
        np.testing.assert_array_equal(new[label], trace[recorded][controls])
    for key in ('actions', 'base_ang_vel', 'dof_pos', 'dof_vel', 'projected_gravity'):
        np.testing.assert_array_equal(new['history_' + key], trace['control_history_' + key][controls])
    np.testing.assert_array_equal(new['teacher_qpos'], trace['qpos'][250:1270])
    np.testing.assert_array_equal(new['teacher_qvel'], trace['qvel'][250:1270])
    np.testing.assert_array_equal(trace['controller_mode'][controls], np.ones(1019))
    np.testing.assert_array_equal(new['joint_limits'], old['joint_limits'])
    np.testing.assert_array_equal(new['joint_span'], old['joint_span'])
    for key in ('feature_mean', 'feature_std'):
        np.testing.assert_array_equal(norm[key], fit[key])
        np.testing.assert_array_equal(norm[key], pairs[key])
    assert np.all(new['expert_target'] >= new['joint_limits'][:, 0])
    assert np.all(new['expert_target'] <= new['joint_limits'][:, 1])
    np.testing.assert_allclose(new['base_target'] + new['residual_rad'], new['expert_target'], atol=3e-16, rtol=0)
    bundle = ROOT / 'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'
    contract_path = bundle / 'contract.json'
    original_path = bundle / 'walk003/original29.npz'
    motion_path = Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910/mjbatch_intent_floor_inputs_v1/walk003/reference.npz')
    contract = json.loads(contract_path.read_text())
    builder = GoalFeatures(read(motion_path), read(original_path), contract)
    default, limits = np.asarray(contract['default_q']), np.asarray(contract['joint_limits'])
    for i, control in enumerate(controls):
        previous = new['previous_action'][i]
        target = np.clip(default + previous * .25 * np.asarray(contract['training_effort']) /
            np.asarray(contract['kp']), limits[:, 0], limits[:, 1])
        feature = np.r_[builder(trace['qpos'][control], trace['qvel'][control], target, control + 11),
                        new['base_target'][i] - default, previous].astype(np.float32)
        np.testing.assert_array_equal(feature, new['features'][i])
    selected = []
    for i, dataset in enumerate((old, query1, new)):
        ids = np.flatnonzero((dataset['control'] >= 250) & (dataset['control'] < 1269))
        np.testing.assert_array_equal(ids, pairs['source_label_row_indices'][i])
        np.testing.assert_array_equal(dataset['control'][ids], controls)
        selected.append({key: dataset[key][ids] for key in ('features', 'expert_target', 'residual_rad')})
    joined = np.concatenate([dataset['features'] for dataset in selected])
    canonical = joined.copy()
    canonical[canonical == 0] = 0.
    groups = {}
    for i, row in enumerate(canonical):
        groups.setdefault(row.tobytes(), []).append(i)
    duplicates = [rows for rows in groups.values() if len(rows) > 1]
    assert len(duplicates) == compatibility['exact_duplicate_groups']
    targets = np.concatenate([dataset['expert_target'] for dataset in selected])
    residuals = np.concatenate([dataset['residual_rad'] for dataset in selected])
    for rows in duplicates:
        np.testing.assert_array_equal(targets[rows], np.broadcast_to(targets[rows[0]], (len(rows), 23)))
        np.testing.assert_array_equal(residuals[rows], np.broadcast_to(residuals[rows[0]], (len(rows), 23)))
    standardized = (joined.astype(float) - norm['feature_mean'].astype(float)) / norm['feature_std'].astype(float)
    names = ('old', 'query1', 'query250')
    checked_pairs, maximum, witnesses = 0, 0., {}
    for a, b in ((0, 1), (0, 2), (1, 2)):
        distance = pairs[names[a] + '__' + names[b] + '_distance']
        assert distance.shape == (1019, 1019) and np.isfinite(distance).all() and np.all(distance >= 0)
        for source, dest, matrix in ((a, b, distance), (b, a, distance.T)):
            name = names[source] + '_to_' + names[dest]
            nearest = np.argmin(matrix, axis=1)
            np.testing.assert_array_equal(nearest, pairs[name + '_nearest_row'])
            minimum = matrix[np.arange(1019), nearest]
            np.testing.assert_array_equal(minimum, pairs[name + '_feature_rms'])
            difference = selected[dest]['expert_target'][nearest] - selected[source]['expert_target']
            np.testing.assert_array_equal(difference, pairs[name + '_target_delta'])
            np.testing.assert_array_equal(selected[dest]['residual_rad'][nearest] - selected[source]['residual_rad'],
                                          pairs[name + '_residual_delta'])
            sensitivity = np.divide(np.sqrt(np.mean(difference**2, axis=1)), minimum,
                out=np.zeros_like(minimum), where=minimum > 0)
            chosen = np.unique(np.r_[np.linspace(0, 1018, 24).astype(int),
                np.argsort(minimum, kind='stable')[:20], np.argsort(sensitivity, kind='stable')[-20:]])
            for i in chosen:
                calculated = np.sqrt(np.mean((standardized[source * 1019 + i] -
                    standardized[dest * 1019:(dest + 1) * 1019])**2, axis=1))
                maximum = max(maximum, float(np.max(np.abs(calculated - matrix[i]))))
                np.testing.assert_allclose(calculated, matrix[i], rtol=2e-14, atol=2e-15)
            checked_pairs += len(chosen) * 1019
            witnesses[name] = chosen.tolist()
    out = BASE / 'bfm250_expert_labels_independent_v1'
    out.mkdir(exist_ok=False)
    report = dict(independent_saved_array_label_checks_pass=True, actual_expert_controls=[250, 1268],
        samples=1019, total_phase_only_training_rows=3057,
        all_actual_targets_histories_prior_integration_bitexact=True, all1069_features_bitexact=True,
        old_normalization_unchanged=True, excluded_initial_standing_and_terminal_BFM=True,
        exact_duplicate_groups=len(duplicates), exact_conflicts=0,
        recomputed_cross_distance_pairs=checked_pairs, maximum_distance_difference=maximum,
        recomputed_rows=witnesses, all_nearest_indices_and_target_residual_deltas_exact=True,
        labels_smoothed_removed_or_averaged=False, physics_steps=0, inference_calls=0, optimizer_calls=0,
        hashes={str(p): sha(p) for p in (*paths.values(), qualification_path, label_report_path,
            compatibility_path, contract_path, original_path, motion_path, Path(__file__),
            ROOT / 'gear_sonic/utils/g1_true23_mpc_student.py')})
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (out / 'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({key: value for key, value in report.items() if key not in ('hashes', 'recomputed_rows')}))


if __name__ == '__main__':
    main()
