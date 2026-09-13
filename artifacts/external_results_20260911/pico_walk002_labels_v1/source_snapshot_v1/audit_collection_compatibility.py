"""Bounded saved-array coverage and exact-conflict report; never fit or infer."""
from pathlib import Path
import numpy as np
from collection_checks import archive, exact, local, read, sha, summary, write


def run(manifest, destination, manifest_sha256):
    destination = Path(destination)
    producer = read(destination / 'rows_complete.json')
    assert producer['complete'] is True and producer['completed_rows'] == 6847
    assert producer['manifest_sha256'] == manifest_sha256
    for case in manifest['cases']:
        report_path = destination / case['clip'] / 'report.json'
        report = read(report_path)
        assert report['complete'] is True and report['samples'] == case['selected_rows']
        assert report['manifest_sha256'] == manifest_sha256
        assert sha(report_path) == producer['case_report_sha256'][case['clip']]
        assert sha(destination / case['clip'] / 'labels.npz') == report['labels_sha256']
    output = destination / 'compatibility'
    output.mkdir(exist_ok=False)
    norm = archive(destination / 'existing_normalization.npz')
    original = archive(local(manifest['normalization']))
    for key in ('feature_mean', 'feature_std'):
        exact(norm[key], original[key], 'original normalization')
    paths = {**{name: local(path) for name, path in manifest['existing_labels'].items()},
             'pico': destination / 'pico/labels.npz', 'walk002': destination / 'walk002/labels.npz'}
    sets = []
    hashes = {}
    limits = span = None
    for name, path in paths.items():
        data = archive(path)
        stop = 6230 if name == 'pico' else 1117 if name == 'walk002' else 1269
        ids = np.flatnonzero((data['control'] >= 250) & (data['control'] < stop))
        exact(data['control'][ids], np.arange(250, stop, dtype=np.int64), 'dataset controls')
        exact(data['source_frame'][ids], np.arange(261, stop+11, dtype=np.int64), 'dataset source frames')
        if limits is None:
            limits, span = data['joint_limits'], data['joint_span']
        exact(data['joint_limits'], limits, 'dataset native limits')
        exact(data['joint_span'], span, 'dataset native span')
        selected = {key: data[key][ids].copy() for key in
                    ('features', 'expert_target', 'residual_rad', 'base_target', 'control', 'source_frame')}
        assert selected['features'].shape == (stop-250, 1069)
        assert selected['features'].dtype == np.float32
        assert all(np.isfinite(value).all() for value in selected.values())
        exact(selected['residual_rad'], selected['expert_target']-selected['base_target'], 'actual residual definition')
        selected['name'] = name
        selected['phase'] = np.where(selected['control'] < 350, 0,
                                     np.where(selected['control'] < stop-100, 1, 2)).astype(np.int64)
        sets.append(selected)
        hashes[name] = sha(path)
    assert [len(data['control']) for data in sets] == [1019,1019,1019,5980,867]
    all_x = np.concatenate([data['features'] for data in sets])
    all_target = np.concatenate([data['expert_target'] for data in sets])
    all_residual = np.concatenate([data['residual_rad'] for data in sets])
    assert all_x.shape == (9904,1069)
    names = [data['name'] for data in sets]
    dataset = np.concatenate([np.full(len(data['control']), i, np.int64) for i,data in enumerate(sets)])
    controls = np.concatenate([data['control'] for data in sets])
    # Numeric duplicate comparison treats signed zeros as equal, on a COPY only.
    canonical = all_x.copy()
    canonical[canonical == 0] = 0.
    groups = {}
    for i, row in enumerate(canonical):
        groups.setdefault(row.tobytes(), []).append(i)
    duplicate = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        target_spread = np.ptp(all_target[indices], axis=0)
        residual_spread = np.ptp(all_residual[indices], axis=0)
        duplicate.append(dict(global_rows=indices, datasets=[names[dataset[i]] for i in indices],
            controls=[int(controls[i]) for i in indices],
            target_spread_by_joint_rad=target_spread.tolist(), residual_spread_by_joint_rad=residual_spread.tolist(),
            target_conflict=bool(np.any(target_spread != 0)), residual_conflict=bool(np.any(residual_spread != 0))))
    old_min = all_x[:3057].min(axis=0)
    old_max = all_x[:3057].max(axis=0)
    reports = {}
    maxima = []
    envelope_counts = []
    for data in sets:
        x = data['features']
        z = (x.astype(np.float64)-norm['feature_mean'].astype(np.float64))/norm['feature_std'].astype(np.float64)
        outside = (x < old_min) | (x > old_max)
        maxima.append(np.max(np.abs(z), axis=0))
        envelope_counts.append(np.sum(outside, axis=0))
        phases = {}
        for code, name in enumerate(('acquisition', 'source', 'return')):
            mask = data['phase'] == code
            phases[name] = dict(rows=int(mask.sum()),
                standardized_feature_RMS=summary(np.sqrt(np.mean(z[mask]**2,axis=1))),
                normalized_residual_RMS=summary(np.sqrt(np.mean((data['residual_rad'][mask]/span)**2,axis=1))))
        reports[data['name']] = dict(rows=len(x), phases=phases,
            standardized_component_abs=summary(np.abs(z)),
            feature_components_abs_z_above6=int(np.sum(np.abs(z) > 6)),
            rows_with_component_abs_z_above6=int(np.sum(np.any(np.abs(z) > 6,axis=1))),
            components_outside_existing3057_feature_envelope=int(outside.sum()),
            rows_outside_existing3057_feature_envelope=int(np.sum(outside.any(axis=1))))
    write(output / 'exact_duplicate_groups.json', duplicate)
    np.savez_compressed(output / 'normalization_coverage.npz', feature_mean=norm['feature_mean'],
        feature_std=norm['feature_std'], dataset_names=np.asarray(names),
        max_abs_standardized_feature_by_dataset=np.stack(maxima),
        outside_old_feature_envelope_counts_by_dataset=np.stack(envelope_counts),
        existing3057_feature_min=old_min, existing3057_feature_max=old_max)
    for name, path in paths.items():
        assert sha(path) == hashes[name]
    report = dict(kind='bounded_five_dataset_saved_array_coverage_and_exact_conflicts',
        total_rows=9904, new_rows=6847, existing_rows=3057, datasets=reports,
        collector_manifest_sha256=manifest_sha256, producer_rows_report_sha256=sha(destination/'rows_complete.json'),
        label_sha256=hashes, normalization_sha256=sha(destination/'existing_normalization.npz'),
        original_normalization_archive_sha256=sha(local(manifest['normalization'])),
        exact_duplicate_groups=len(duplicate),
        exact_target_conflict_groups=sum(d['target_conflict'] for d in duplicate),
        exact_residual_conflict_groups=sum(d['residual_conflict'] for d in duplicate),
        normalization_refitted=False, duplicates_averaged_or_removed=False,
        signed_zeros_canonicalized_only_in_duplicate_comparison_copy=True,
        nearest_neighbor_distances_computed=False, full_compatibility_or_training_qualification=False,
        inference_calls=0, physics_steps=0, optimizer_calls=0, fitting_launched=False,
        coverage_sha256=sha(output/'normalization_coverage.npz'),
        duplicate_groups_sha256=sha(output/'exact_duplicate_groups.json'))
    write(output / 'report.json', report)
    return report
