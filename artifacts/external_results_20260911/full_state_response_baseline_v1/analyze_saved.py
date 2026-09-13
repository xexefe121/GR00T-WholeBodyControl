"""Compare saved finite-feedback responses against zero response; no model calls."""
import hashlib
import json
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent
NEW = BASE.parent
FIT = NEW / 'direct_target_full_state_student_v1/fit'
DATA = NEW / 'direct_target_full_state_secants_v1/generation'
PINS = {}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def bind(path, expected=None):
    digest = sha(path)
    if expected is not None:
        assert digest == expected, str(path)
    PINS[Path(path).as_posix()] = digest
    return path

def read(path, expected=None):
    return json.loads(Path(bind(path, expected)).read_text(encoding='utf-8-sig'))

def run():
    manifest = read(DATA / 'manifest.json', '2c337d78ca7c9d291ab0e07f95fa34a3175c9e245aef4ff79303b6e57f3a4638')
    outputs = read(FIT / 'output_manifest.json', '290d72143d393728cd7bef4a7283c9fe8a9ab2960c3c0347a2eb2be56c235cc2')
    report = read(FIT / 'report.json', '7426d281be79c3610ef02ae8889c3ef66264da15a9667551d6bca4166a4beaea')
    with np.load(bind(DATA / 'centers.npz', manifest['centers_sha256']), allow_pickle=False) as data:
        centers = data['target'].copy()
        dataset = data['dataset'].copy()
        phase = data['phase'].copy()
    with np.load(bind(FIT / 'normalization.npz', outputs['files']['normalization.npz']), allow_pickle=False) as data:
        span = data['joint_span'].astype(np.float64)
    teacher = np.load(bind(DATA / manifest['arrays']['target']['path'], manifest['arrays']['target']['sha256']), mmap_mode='r')
    wanted = (teacher - centers[:, None, None, :]) / span
    assert wanted.shape == (3057, 58, 2, 23) and np.isfinite(wanted).all()
    sizes = (3, 3, 23, 3, 3, 23)
    labels = ('root_position', 'root_rotation', 'joint_position', 'root_linear_velocity', 'root_angular_velocity', 'joint_velocity')
    groups = np.repeat(np.arange(6), sizes)
    masks = [(dataset == d) & (phase == p) for d in range(3) for p in range(3)]

    def summarize(error):
        odd = (error[:, :, 1, :] - error[:, :, 0, :]) * .5
        even = (error[:, :, 1, :] + error[:, :, 0, :]) * .5
        cells = []
        for c, mask in enumerate(masks):
            for g in range(6):
                selected = groups == g
                total = float(np.square(error[mask][:, selected]).mean())
                odd_mse = float(np.square(odd[mask][:, selected]).mean())
                even_mse = float(np.square(even[mask][:, selected]).mean())
                assert np.isclose(total, odd_mse + even_mse, rtol=1e-12, atol=1e-16)
                cells.append(dict(dataset=c // 3, phase=c % 3, group=labels[g], total=total, odd=odd_mse, even=even_mse))
        return dict(objective=float(np.mean([c['total'] for c in cells])),
            odd=float(np.mean([c['odd'] for c in cells])), even=float(np.mean([c['even'] for c in cells])), cells=cells)

    results = {'zero_response': summarize(-wanted)}
    for label in ('initial_GPU32', 'final_GPU32'):
        def array(corpus):
            name = label + '_' + corpus + '.npy'
            return np.load(bind(FIT / name, outputs['files'][name]), mmap_mode='r')
        nominal = array('nominal')[:3057].astype(np.float64)
        endpoint = array('full_state').astype(np.float64).reshape(3057, 58, 2, 23)
        results[label] = summarize(endpoint - nominal[:, None, None, :] - wanted)
        assert np.isclose(results[label]['objective'], report['metrics'][label]['full_state_objective'], rtol=1e-12, atol=1e-16)
    comparison = []
    for g, label in enumerate(labels):
        row = dict(group=label)
        for backend, result in results.items():
            row[backend] = float(np.mean([c['total'] for c in result['cells'] if c['group'] == label]))
        row['final_over_zero'] = row['final_GPU32'] / row['zero_response'] if row['zero_response'] else None
        comparison.append(row)
    for path, digest in PINS.items():
        assert sha(path) == digest
    result = dict(saved_analysis_complete=True, comparison=comparison, results=results, input_sha256=PINS,
        source_sha256=sha(__file__), model_calls=0, native_steps=0, optimizer_updates=0,
        limitations=['Finite changes normalized by joint target span; physical radii are not divided out.',
            'Zero response holds each predicted center output across its paired perturbations; it is not a controller.',
            'Odd and even paired-error decomposition does not establish closed-loop stability or fresh expert behavior.'])
    with (BASE / 'report.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(objectives={k: v['objective'] for k, v in results.items()}, comparison=comparison,
        final_odd_fraction=results['final_GPU32']['odd'] / results['final_GPU32']['objective'], report_sha256=sha(BASE / 'report.json'))))

if __name__ == '__main__':
    run()
