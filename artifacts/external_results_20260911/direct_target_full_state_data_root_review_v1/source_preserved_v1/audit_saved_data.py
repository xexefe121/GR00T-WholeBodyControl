"""Full saved-data qualification. Actual model/native/optimizer calls remain zero."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from saved_math import (GROUPS, KEPT, original_difference, expected_radii, exact,
                        check_probe, map_values, duplicate_conflicts)

def local(path):
    text = str(path).replace('\\', '/')
    if sys.platform != 'win32' and len(text) > 2 and text[1] == ':':
        text = '/mnt/' + text[0].lower() + text[2:]
    return Path(text)

def sha(path):
    h = hashlib.sha256()
    with local(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(local(path).read_text(encoding='utf-8-sig'))

def archive(path):
    with np.load(local(path), allow_pickle=False) as values:
        return {key: values[key].copy() for key in values.files}

def run(request_path, generation, training_request_path, original_feature_source, output):
    output.mkdir(exist_ok=False)
    tracked = {}
    checks = 0
    completed_rows = 0
    current = None
    def bind(path, expected=None):
        path = local(path)
        digest = sha(path)
        if expected is not None and digest != expected:
            raise AssertionError('Subject hash: ' + str(path))
        if str(path) in tracked and tracked[str(path)] != digest:
            raise AssertionError('Changed subject: ' + str(path))
        tracked[str(path)] = digest
        return path
    def check(condition, name):
        nonlocal checks
        checks += 1
        if not condition:
            raise AssertionError(name)
    try:
        for source in (Path(__file__), Path(__file__).with_name('saved_math.py')):
            bind(source)
        request = read(bind(request_path))
        report = read(bind(generation / 'report.json'))
        check(request['generation_selected'] is True, 'Selected generation')
        check(request['rows'] == 3057 and request['axes'] == 58 and request['signed_rows'] == 354612, 'Original full scope')
        check(report['complete'] is True and report['request_sha256'] == sha(request_path), 'Actual complete report/request')
        check(report['original_center_and_overlap_exact'] is True and report['all_slots_valid'] is True, 'Producer complete gates')
        for key in ('model_calls', 'BFM_calls', 'physics_steps', 'optimizer_updates', 'replans'):
            check(report[key] == 0, 'Pure generation: ' + key)
        for path, digest in request['input_sha256'].items():
            bind(path, digest)
        for path, digest in request['source_sha256'].items():
            bind(path, digest)
        paths = {key: local(value) for key, value in request['paths'].items()}
        pinned_inputs = {local(key).resolve() for key in request['input_sha256']}
        check(all(path.resolve() in pinned_inputs for path in paths.values()), 'All consumed input roles are pinned')
        for name, digest in report['output_sha256'].items():
            check(Path(name).name == name, 'Output basename')
            bind(generation / name, digest)
        centers = archive(paths['centers'])
        saved_centers = archive(generation / 'centers.npz')
        contract = read(paths['contract'])
        limits = np.asarray(contract['joint_limits'], np.float64)
        caps = np.asarray(contract['native_velocity'], np.float64)
        default = np.asarray(contract['default_q'], np.float64)
        spans = centers['joint_span'].astype(np.float64)
        exact(centers['joint_limits'], limits, 'Original native limits')
        exact(centers['joint_span'], (limits[:, 1]-limits[:, 0]).astype(np.float32), 'Original rounded native span')
        radii = expected_radii(caps)
        for name in ('qpos', 'qvel', 'dataset', 'control', 'source_frame', 'joint_span', 'joint_limits'):
            exact(saved_centers[name], centers[name], 'Saved center ' + name)
        exact(saved_centers['features'], centers['features'][:, KEPT], 'All nominal retained features')
        exact(saved_centers['target'], centers['expert_target'], 'All nominal teacher targets')
        exact(saved_centers['axis_radius'], radii, 'All fixed physical radii')
        group_ids = np.concatenate([np.full(b-a, i, np.int8) for i, (a, b) in enumerate(GROUPS)])
        exact(saved_centers['axis_group'], group_ids, 'Six tangent groups')
        expected_phase = np.where(centers['control'] < 350, 0, np.where(centers['control'] < 1169, 1, 2)).astype(np.int8)
        expected_cell = (centers['dataset'] * 3 + expected_phase).astype(np.int8)
        exact(saved_centers['phase'], expected_phase, 'Three original phases')
        exact(saved_centers['cell'], expected_cell, 'Nine original cells')
        check(np.array_equal(np.bincount(expected_cell), [100, 819, 100]*3), 'Requested cell counts')
        # Byte-preserved earlier feature builder is already qualified independently.
        bind(original_feature_source, '26b816a2059edbb83daba386196c21b70002d1c83e2eda6e5cbf6a550ee4a251')
        spec = importlib.util.spec_from_file_location('original_pure_direct_features', original_feature_source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        builder = module.DirectFeatures(archive(paths['motion']), archive(paths['original29']), contract)
        difference = original_difference(paths['core'])
        arrays = {name: np.load(generation / (name + '.npy'), mmap_mode='r', allow_pickle=False)
                  for name in ('features', 'target', 'target_change', 'qpos', 'qvel', 'tangent', 'tangent_change',
                      'feedback_raw', 'feedback_correction', 'preclip', 'feedback_clipped', 'native_clipped', 'signed_radius', 'status')}
        shapes = {'features': (1000, np.float32), 'target': (23, np.float64), 'target_change': (23, np.float64),
            'qpos': (30, np.float64), 'qvel': (29, np.float64), 'tangent': (58, np.float64), 'tangent_change': (58, np.float64),
            'feedback_raw': (23, np.float64), 'feedback_correction': (23, np.float64), 'preclip': (23, np.float64),
            'feedback_clipped': (23, np.bool_), 'native_clipped': (23, np.bool_), 'signed_radius': (None, np.float64), 'status': (None, np.uint8)}
        for name, (width, dtype) in shapes.items():
            expected_shape = (3057, 58, 2) + (() if width is None else (width,))
            check(arrays[name].shape == expected_shape and arrays[name].dtype == dtype, 'Schema: ' + name)
            check(name + '.npy' in report['output_sha256'], 'Output schema bound: ' + name)
        check(np.all(arrays['status'] == 1), 'All354612 slots committed')
        old = {key: np.load(paths[key], mmap_mode='r', allow_pickle=False) for key in
               ('velocity_features', 'velocity_target', 'velocity_raw', 'velocity_feedback_clipped',
                'velocity_native_clipped', 'velocity_value')}
        group_counts = np.zeros((9, 6), np.int64)
        group_feedback_clips = np.zeros((9, 6), np.int64)
        group_native_clips = np.zeros((9, 6), np.int64)
        group_zero_response = np.zeros((9, 6), np.int64)
        new_feature_recomputations = 0
        for i in range(3057):
            q0, v0 = centers['qpos'][i], centers['qvel'][i]
            plan = centers['planned_state'][i]
            for axis in range(58):
                for sign_index, sign in enumerate((-1., 1.)):
                    at = (i, axis, sign_index)
                    current = dict(center=i, axis=axis, sign_index=sign_index)
                    q, v = arrays['qpos'][at], arrays['qvel'][at]
                    displacement = sign * radii[axis]
                    exact(arrays['signed_radius'][at], np.float64(displacement), 'Signed physical radius')
                    tangent, tangent_change = check_probe(q0, v0, q, v, axis, displacement, plan, difference, limits, caps)
                    exact(arrays['tangent'][at], tangent, 'Full58 saved tangent')
                    exact(arrays['tangent_change'][at], tangent_change, 'Full58 tangent change')
                    if axis >= 35:
                        old_at = (i, axis-35, sign_index)
                        feature = old['velocity_features'][old_at][KEPT]
                        raw = old['velocity_raw'][old_at]
                        correction = np.clip(raw, -.1, .1)
                        preclip = centers['planned_target'][i] + correction
                        target = old['velocity_target'][old_at]
                        exact(arrays['feedback_clipped'][at], old['velocity_feedback_clipped'][old_at], 'Old feedback clipping overlap')
                        exact(arrays['native_clipped'][at], old['velocity_native_clipped'][old_at], 'Old native clipping overlap')
                        exact(v[6+axis-35], old['velocity_value'][old_at], 'Old perturbed velocity overlap')
                    else:
                        feature = builder(q, v, int(centers['source_frame'][i]))
                        raw, correction, preclip, target = map_values(centers['gain'][i], tangent, centers['planned_target'][i], limits)
                        new_feature_recomputations += 1
                    for key, value in [('features', feature), ('target', target), ('feedback_raw', raw),
                        ('feedback_correction', correction), ('preclip', preclip),
                        ('feedback_clipped', raw != correction), ('native_clipped', preclip != target),
                        ('target_change', target-centers['expert_target'][i])]:
                        exact(arrays[key][at], value, 'Saved ' + key)
                    check(np.isfinite(feature).all() and np.isfinite(target).all(), 'Finite returned feature/target')
                    check(np.all(target >= limits[:, 0]) and np.all(target <= limits[:, 1]), 'Native target bounds')
                    check(not np.array_equal(feature, saved_centers['features'][i]), 'No nominal feature alias')
                    if sign_index:
                        check(not np.array_equal(feature, arrays['features'][i, axis, 0]), 'No opposite-sign alias')
                    cell, group = int(expected_cell[i]), int(group_ids[axis])
                    group_counts[cell, group] += 1
                    group_feedback_clips[cell, group] += int(np.any(raw != correction))
                    group_native_clips[cell, group] += int(np.any(preclip != target))
                    group_zero_response[cell, group] += int(np.array_equal(target, centers['expert_target'][i]))
                    completed_rows += 1
            if (i+1) % 100 == 0:
                temporary = output / 'progress.tmp'
                temporary.write_text(json.dumps(dict(centers_checked=i+1, rows_checked=completed_rows)) + '\n')
                temporary.replace(output / 'progress.json')
        expected_counts = np.array([100, 819, 100]*3)[:, None] * np.array([3, 3, 23, 3, 3, 23])[None, :] * 2
        check(np.array_equal(group_counts, expected_counts), 'Every requested54cell count')
        check(completed_rows == 354612 and new_feature_recomputations == 213990, 'Complete full-state and new-row counts')
        training_request = read(bind(training_request_path, 'e3dbc8c41433699326cd611ea97767974b6eb9ce5de2c7b1f678f83594054625'))
        training_frozen_path = training_request_path.parent / 'training_frozen_inputs.json'
        training_frozen = read(bind(training_frozen_path, '1b9499a88b26b9feb5e1259d63284010e7521150c591a9815063cacc0251fa9c'))
        check(training_frozen['training_request_sha256'] == sha(training_request_path), 'Original anchor request/frozen linkage')
        anchor_pins = {str(local(path).resolve()): digest for path, digest in training_frozen['input_sha256'].items()}
        def anchor(path):
            path = local(path)
            return bind(path, anchor_pins[str(path.resolve())])
        anchor_paths = training_request['paths']
        pico = archive(anchor(anchor_paths['pico']))
        walk = archive(anchor(anchor_paths['walk002']))
        nominal_features = np.concatenate([saved_centers['features'], pico['features'][:, KEPT], walk['features'][:, KEPT]])
        nominal_targets = np.concatenate([saved_centers['target'], pico['expert_target'], walk['expert_target']])
        check(nominal_features.shape == (9904, 1000) and nominal_features.dtype == np.float32, 'All9904 nominal anchors')
        manifest_path = anchor(anchor_paths['physical_manifest'])
        manifest = read(manifest_path)
        def physical(name):
            item = manifest['arrays'][name]
            path = (manifest_path.parent / item['path']).resolve()
            check(path.parent == manifest_path.parent.resolve(), 'Physical manifest local member')
            anchor(path)
            bind(path, item['sha256'])
            result = np.load(path, mmap_mode='r', allow_pickle=False)
            check(list(result.shape) == item['shape'] and str(result.dtype) == item['dtype'], 'Physical array schema')
            return result
        physical_features = physical('endpoint_features')[:, KEPT]
        physical_targets = physical('label_fixed_map_target')
        check(physical_features.shape == (3054, 1000) and physical_targets.shape == (3054, 23), 'All3054 physical anchors')
        index, duplicates, conflicts = {}, [], []
        corpus_sizes = {}
        for name, features, targets in (
            ('nominal', nominal_features, nominal_targets),
            ('full_state', arrays['features'].reshape(354612, 1000), arrays['target'].reshape(354612, 23)),
            ('physical', physical_features, physical_targets)):
            found, incompatible = duplicate_conflicts(features, targets, spans, default, index, name)
            duplicates.extend(found)
            conflicts.extend(incompatible)
            corpus_sizes[name] = len(features)
        duplicate_report = dict(corpus_sizes=corpus_sizes, total_rows=sum(corpus_sizes.values()),
            unique_numerical_feature_rows=len(index), duplicate_rows=len(duplicates),
            incompatible_normalized_float32_target_rows=len(conflicts),
            duplicates=duplicates, conflicts=conflicts,
            convention='Finite float32 features; signed zero canonicalized. Same normalized-float32 label is representable by the head; raw-float64 label differences remain reported. No rows removed.')
        (output / 'duplicate_conflicts.json').write_text(json.dumps(duplicate_report, indent=2, allow_nan=False) + '\n')
        check(sum(corpus_sizes.values()) == 367570, 'Full intended training population')
        check(not conflicts, 'No incompatible exact-input training labels')
        check(all(sha(path) == digest for path, digest in tracked.items()), 'All subjects unchanged after audit')
        result = dict(data_review_pass=True, request_sha256=sha(request_path), generation_report_sha256=sha(generation/'report.json'),
            checks=checks, rows_checked=completed_rows, exact_old_overlap_rows=140622,
            independently_recomputed_new_features_and_maps=new_feature_recomputations,
            full_tangent_and_state_rows=completed_rows, original_native_bounds_unchanged=True,
            complete54cell_counts=group_counts.tolist(), feedback_clipped_rows=group_feedback_clips.tolist(),
            native_clipped_rows=group_native_clips.tolist(), zero_target_response_rows=group_zero_response.tolist(),
            original_zero_gain_centers=int(np.count_nonzero(~np.any(centers['gain'] != 0, axis=(1, 2)))),
            corpus_sizes=corpus_sizes, unique_numerical_feature_rows=len(index), duplicate_rows=len(duplicates),
            incompatible_normalized_float32_target_rows=len(conflicts),
            duplicate_report_sha256=sha(output/'duplicate_conflicts.json'), input_sha256=tracked,
            model_calls=0, BFM_calls=0, native_steps=0, optimizer_updates=0, replans=0,
            limitation='Checks fixed saved-map data, not contact consistency, dynamic recovery, sufficient state coverage or closed-loop stability. Only a future fit and unchanged full physical tests can establish controller behavior.')
        (output / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print(json.dumps(dict(passed=True, rows=completed_rows, new_recomputations=new_feature_recomputations,
            unique_rows=len(index), duplicates=len(duplicates), conflicts=len(conflicts))))
    except BaseException as error:
        (output / 'failure.json').write_text(json.dumps(dict(error=repr(error), checks=checks,
            rows_checked=completed_rows, active=current, input_sha256=tracked,
            model_calls=0, native_steps=0, optimizer_updates=0), indent=2) + '\n')
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for flag in ('request', 'generation', 'training-request', 'original-feature-source', 'output'):
        parser.add_argument('--' + flag, required=True)
    args = parser.parse_args()
    run(local(args.request), local(args.generation), local(args.training_request),
        local(args.original_feature_source), local(args.output))
