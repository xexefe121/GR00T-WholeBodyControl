"""One bounded saved-array assessment. No model, optimizer, or simulator imports."""
from pathlib import Path
import hashlib
import json
import sys
import traceback
import numpy as np
from feature_scan import Groups, reduced

NEW = Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')
HERE = Path(__file__).resolve().parent.parent
GEN = NEW/'velocity_chord_student_v1/generation'
DATA = NEW/'one_step_policy_branch_collection_resume2969_v1/collection/data'
SNAP = NEW/'one_step_physical_student_v1/source_snapshot_v1'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    out = HERE/'results'
    out.mkdir(exist_ok=False)
    pins = {}
    def pin(path, expected=None):
        path = Path(path).resolve(); value = sha(path)
        if expected is not None and value != expected: raise ValueError(f'pin mismatch: {path}')
        pins[str(path)] = value
        return path
    def read(path, expected=None):
        return json.loads(pin(path, expected).read_text(encoding='utf-8-sig'))
    try:
        for path in sorted((HERE/'source').glob('*.py')): pin(path)
        pin(HERE/'tests.txt')
        for name in ('student_linear_runtime.py', 'gear_sonic/utils/g1_true23_mpc_student.py',
                     'gear_sonic/utils/g1_true23_bfm_seed_observations.py'):
            pin(SNAP/name)
        generation = read(GEN/'report.json')
        velocity_review = read(NEW/'velocity_chord_completed_data_review_v1/review.json',
                              'c85e2e9a29007a8c80f496fe567098e08f1d6e397f72af71e3f56cf4658a305c')
        assert velocity_review['passed'] and generation['complete'] and generation['sign_order'] == [-1, 1]
        assert sha(GEN/'report.json') in velocity_review['evidence_hashes'].values()
        centers = np.load(pin(GEN/'centers.npz', velocity_review['output_hashes']['centers.npz']), allow_pickle=False)
        vf = np.load(pin(GEN/'features.npy', velocity_review['output_hashes']['features.npy']), mmap_mode='r', allow_pickle=False)
        vt = np.load(pin(GEN/'teacher_target.npy', velocity_review['output_hashes']['teacher_target.npy']), mmap_mode='r', allow_pickle=False)
        assert vf.shape == (3057,23,2,1069) and vt.shape == (3057,23,2,23)
        assert centers['features'].shape == (3057,1069) and centers['expert_target'].shape == (3057,23)
        assert np.array_equal(centers['dataset'], np.repeat(np.arange(3, dtype=np.int64), 1019))
        assert np.array_equal(centers['control'], np.tile(np.arange(250,1269,dtype=np.int64), 3))
        assert np.array_equal(centers['source_frame'], centers['control'] + 11)
        limits = centers['joint_limits']
        assert limits.shape == (23,2) and limits.dtype == np.float64
        branch_review = read(NEW/'one_step_branch_combined_data_review_v1/review.json',
                             '6faf42e89281ac155fdc35a6b38f5b5218c2d067318c46b7a883cafc1797d9ed')
        assert branch_review['data_review_pass']
        manifest = read(DATA/'manifest.json', '92fa1575670b33c361d909d76cf8a892f1cdb9c420949e63b2040c6e1dd01260')
        assert pins[str((DATA/'manifest.json').resolve())] in json.dumps(branch_review)
        branch_report = read(DATA.parent/'report.json', 'caf8342428008bf1b4551a0c3ee84cadc6d3b401de479a9ec7a2c7a273802fce')
        def branch(key):
            spec = manifest['arrays'][key]
            arr = np.load(pin(DATA/spec['path'], spec['sha256']), mmap_mode='r', allow_pickle=False)
            assert list(arr.shape) == spec['shape'] and str(arr.dtype) == spec['dtype']
            return arr
        bf, bt = branch('endpoint_features'), branch('label_fixed_map_target')
        assert bf.shape == (3054,1069) and bt.shape == (3054,23)
        expected_dataset = np.repeat(np.arange(3, dtype=np.int64), 1018)
        expected_start = np.tile(np.arange(250,1268,dtype=np.int64),3)
        for key, expected in [('dataset', expected_dataset), ('start_control', expected_start),
                              ('successor_control', expected_start+1),
                              ('center_index', expected_dataset*1019+expected_start-250),
                              ('successor_center_index', expected_dataset*1019+expected_start-249)]:
            actual = branch(key)
            assert actual.dtype == np.int64 and actual.tobytes() == expected.tobytes()
        assert branch('label_valid').dtype == np.bool_ and branch('label_valid').all()
        sources = [('nominal', centers['features'], centers['expert_target']),
                   ('velocity', vf.reshape(-1,1069), vt.reshape(-1,23)), ('physical', bf, bt)]
        broader_request = read(NEW/'broader_labels_independent_v1/inference_request_v2.json')
        broader_pins = {str(Path(k).resolve()):v for k,v in broader_request['input_hashes'].items()}
        broader_saved = read(NEW/'broader_labels_independent_v1/saved_array_audit/report.json',
                             broader_pins[str((NEW/'broader_labels_independent_v1/saved_array_audit/report.json').resolve())])
        broader_inference = read(NEW/'broader_labels_independent_v1/baseline_inference_audit/report.json')
        assert sha(NEW/'broader_labels_independent_v1/inference_request_v2.json') in json.dumps(broader_inference)
        assert broader_saved['pass_all'] and broader_inference['pass_all']
        assert broader_saved['selected_rows_checked'] == broader_inference['selected_rows_checked'] == 6847
        # Existing unknown wrapper exit is retained as historical evidence, not changed to zero.
        pin(NEW/'pico_walk002_labels_v1/completion_verification.json')
        for clip, stop in [('pico',6230), ('walk002',1117)]:
            directory = NEW/f'pico_walk002_labels_v1/collection/{clip}'
            report = read(directory/'report.json')
            labels = np.load(pin(directory/'labels.npz', report['labels_sha256']), allow_pickle=False)
            assert report['labels_sha256'] == broader_pins[str((directory/'labels.npz').resolve())]
            assert report['complete'] and report['samples'] == stop-250
            assert labels['control'].dtype == np.int64
            assert labels['control'].tobytes() == np.arange(250,stop,dtype=np.int64).tobytes()
            assert np.array_equal(labels['source_frame'], labels['control']+11)
            assert labels['joint_limits'].dtype == limits.dtype and labels['joint_limits'].tobytes() == limits.tobytes()
            assert labels['features'].shape == (stop-250,1069)
            sources.append((clip, labels['features'], labels['expert_target']))
        starts = np.cumsum([0]+[len(s[1]) for s in sources])
        assert starts.tolist() == [0,3057,143679,146733,152713,153580]
        def row_reader(index):
            source = int(np.searchsorted(starts, index, side='right')-1)
            local = index-int(starts[source])
            return sources[source][1][local], sources[source][2][local]
        for name, features, targets in sources:
            assert features.dtype == np.float32 and targets.dtype == np.float64
            assert np.isfinite(features).all() and np.isfinite(targets).all(), name
        request = dict(kind='saved_absolute_target_reduced_feature_feasibility_request',
                       input_source_sha256=pins, mandatory_rows=146733, optional_rows=6847,
                       retained_slices=[[0,52],[75,1023]], removed_slices=[[52,75],[1023,1069]],
                       input_dtype='float32', target_dtype='float64', target='absolute native position, never residual',
                       source_order=[dict(name=n, rows=len(x), feature_shape=list(x.shape), target_shape=list(y.shape)) for n,x,y in sources],
                       inference_calls=0, physics_steps=0, optimizer_updates=0, labels_generated=0)
        dump(HERE/'request.json', request)
        exact = Groups(row_reader); numeric = Groups(row_reader, numeric_zero=True)
        total = int(starts[-1]); digests = np.empty((total,32), np.uint8); numeric_digests = np.empty_like(digests)
        target_digests = np.empty_like(digests); groups = np.empty(total,np.int64); numeric_groups = np.empty_like(groups)
        stats = []; mandatory = None
        for source_index, (name, features, targets) in enumerate(sources):
            zero_count = negative_zero_count = target_negative_zero = 0
            for local in range(len(features)):
                row_id = int(starts[source_index])+local
                x, y = features[local], targets[local]
                value = reduced(x)
                zero_count += int(np.count_nonzero(value == 0))
                negative_zero_count += int(np.count_nonzero((value == 0) & np.signbit(value)))
                target_negative_zero += int(np.count_nonzero((y == 0) & np.signbit(y)))
                digest, groups[row_id] = exact.add(row_id, x, y)
                numeric_digest, numeric_groups[row_id] = numeric.add(row_id, x, y)
                digests[row_id] = np.frombuffer(digest,np.uint8)
                numeric_digests[row_id] = np.frombuffer(numeric_digest,np.uint8)
                target_digests[row_id] = np.frombuffer(hashlib.sha256(y.tobytes()).digest(),np.uint8)
            outside = (targets < limits[:,0]) | (targets > limits[:,1])
            at_lower, at_upper = targets == limits[:,0], targets == limits[:,1]
            quantized = targets.astype(np.float32).astype(np.float64)
            stats.append(dict(name=name, rows=len(features), absolute_target_min=targets.min(0).tolist(),
                              absolute_target_max=targets.max(0).tolist(), outside_native_bound_components=int(outside.sum()),
                              outside_native_bound_rows=int(np.any(outside,axis=1).sum()),
                              at_native_lower_components=int(at_lower.sum()), at_native_upper_components=int(at_upper.sum()),
                              retained_zero_components=zero_count, retained_negative_zero_components=negative_zero_count,
                              absolute_target_negative_zero_components=target_negative_zero,
                              float32_output_roundtrip_max_abs_rad=float(np.max(np.abs(quantized-targets))),
                              float32_roundtrip_nonexact_components=int(np.count_nonzero(quantized != targets)),
                              float32_roundtrip_outside_native_bound_components=int(((quantized<limits[:,0])|(quantized>limits[:,1])).sum())))
            if name == 'physical': mandatory = dict(byte_exact=exact.summary(), numerical_zero_equivalence=numeric.summary())
            print(name, len(features), exact.summary(), flush=True)
        np.savez_compressed(out/'row_identity_hashes.npz', source_starts=starts,
                            reduced_input_sha256=digests, numerical_zero_key_sha256=numeric_digests,
                            absolute_target_sha256=target_digests, group=groups, numerical_zero_group=numeric_groups)
        conflicts = dict(byte_exact_target_byte_pairs=exact.target_byte_conflicts,
                         byte_exact_target_numeric_pairs=exact.target_numeric_conflicts,
                         numerical_zero_target_byte_pairs=numeric.target_byte_conflicts,
                         numerical_zero_target_numeric_pairs=numeric.target_numeric_conflicts)
        dump(out/'conflicts.json', conflicts)
        final_hashes = {path:sha(path) for path in pins}
        assert final_hashes == pins
        report = dict(kind='pure_saved_array_direct_absolute_target_feature_feasibility', completed=True,
                      request_sha256=sha(HERE/'request.json'), mandatory=mandatory,
                      expanded=dict(byte_exact=exact.summary(), numerical_zero_equivalence=numeric.summary()),
                      row_identity_conventions=dict(nominal='dataset=floor(i/1019), control=250+i%1019, source=control+11',
                        velocity='center=floor(i/46), axis=floor(i%46/2), sign=[-1,+1][i%2]; same center clock',
                        physical='dataset=floor(i/1018), start=250+i%1018, successor=start+1, source=successor+11',
                        pico='control=250+i, source=control+11', walk002='control=250+i, source=control+11'),
                      per_source=stats, native_joint_limits=limits.tolist(),
                      all_initial_final_pins_exact=True, pin_count=len(pins),
                      output_sha256={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()},
                      no_arrays_changed=True, no_rows_dropped=True, no_targets_averaged=True,
                      model_calls=0, physics_steps=0, optimizer_updates=0, labels_generated=0,
                      runtime_python=sys.version, runtime_numpy=np.__version__,
                      limitations=['Exact duplicate screening is not a test of nearby consistency, Markov sufficiency, balance, timing, or recovery.',
                                   'Velocity probes are fixed committed-map labels, not new physical trajectories.',
                                   'Numerical-zero equivalence uses separate scratch keys only; original signs/dtypes/bytes are retained.',
                                   'Historical broader producer wrapper unknown exit remains unknown; completed data was independently qualified.',
                                   'No architecture, fitting, new normalization, inference, or controller change selected.'])
        dump(out/'report.json', report)
    except BaseException as exc:
        dump(out/'failure.json', dict(kind=type(exc).__name__, error=str(exc), traceback=traceback.format_exc(), input_pins=pins))
        raise


if __name__ == '__main__': main()
