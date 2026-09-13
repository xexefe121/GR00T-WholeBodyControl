"""Inspect saved gradient vectors only. Never constructs or calls a model."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')
DATA = ROOT / 'direct_target_fp64_followup_analysis_v1/gradient_results'
OUT = Path(__file__).parent
SHAPES = {'actor.0.weight': (256, 1000), 'actor.0.bias': (256,),
          'actor.2.weight': (256, 256), 'actor.2.bias': (256,),
          'actor.4.weight': (23, 256), 'actor.4.bias': (23,)}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    destination = OUT / 'report.json'
    if destination.exists():
        raise FileExistsError('Preserve prior saved review.')
    report_path = DATA / 'report.json'
    owner = json.loads(report_path.read_text())
    assert owner['passed'] and owner['parameter_state_unchanged']
    assert owner['optimizer_updates'] == owner['native_calls'] == 0
    assert owner['counters'] == dict(forward_attempted=3, forward_returned=3,
        forward_rows_attempted=14110, forward_rows_returned=14110,
        gradient_attempted=3, gradient_returned=3)
    assert owner['parameter_names'] == list(SHAPES)
    pins = {str(report_path): sha(report_path)}
    vectors = {}
    for loss in ('nominal', 'velocity', 'physical'):
        path = DATA / (loss + '_gradients.npz')
        pins[str(path)] = sha(path)
        assert owner['output_sha256'][path.name] == sha(path)
        with np.load(path, allow_pickle=False) as saved:
            assert saved.files == list(SHAPES)
            for name, shape in SHAPES.items():
                assert saved[name].shape == shape and saved[name].dtype == np.float32
                assert np.isfinite(saved[name]).all()
            vectors[loss] = np.concatenate([saved[n].astype(np.float64).ravel() for n in SHAPES])
    names = list(vectors)
    gram = np.array([[np.sum(vectors[a] * vectors[b], dtype=np.float64)
                      for b in names] for a in names])
    norms = np.sqrt(np.diag(gram))
    cosine = gram / (norms[:, None] * norms[None, :])
    for i, name in enumerate(names):
        np.testing.assert_allclose(norms[i], owner['parameter_gradient_norms'][name], rtol=1e-12, atol=0)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        np.testing.assert_allclose(cosine[i, j], owner['pairwise_cosines'][names[i]+'_'+names[j]], rtol=1e-12, atol=1e-15)
    # G = g_nominal + weight*g_velocity + g_physical.
    # Positive gi dot G means a sufficiently small -G step decreases loss i.
    anchor = gram[:, 0] + gram[:, 2]
    slope = gram[:, 1]
    lower, upper = 0., float('inf')
    inequalities = {}
    feasible = True
    for i, name in enumerate(names):
        a, b = float(anchor[i]), float(slope[i])
        inequalities[name] = dict(constant=a, coefficient=b)
        if b > 0:
            lower = max(lower, -a / b)
        elif b < 0:
            upper = min(upper, -a / b)
        elif a < 0:
            feasible = False
    feasible = feasible and lower <= upper
    balanced = float(norms[0] / norms[1])
    compositions = {}
    for label, weight in [('original', 1.), ('literal_secant', 10000.), ('equal_nominal_gradient', balanced)]:
        coeff = np.array([1., weight, 1.])
        magnitude = float(np.sqrt(coeff @ gram @ coeff))
        compositions[label] = dict(velocity_weight=weight, total_norm=magnitude,
            descent_dot_per_loss=dict(zip(names, (gram @ coeff).tolist())),
            cosine_nominal=float((gram @ coeff)[0]/(magnitude*norms[0])))
    result = dict(saved_gradient_math_passed=True, input_sha256=pins,
        source_sha256=sha(Path(__file__)), names=names, gram_matrix=gram.tolist(),
        norms=dict(zip(names, norms.tolist())), cosine_matrix=cosine.tolist(),
        compositions=compositions,
        first_order_common_descent_interval=dict(feasible=feasible, minimum_weight=lower,
            maximum_weight=upper if np.isfinite(upper) else None, inequalities=inequalities),
        limitation='One fixed sampled velocity batch at current weights; no finite-step, later-update or full-trajectory guarantee. This checks saved vector arithmetic, not autograd itself.',
        model_calls=0, native_steps=0, optimizer_updates=0,
        selected_weight=None)
    assert all(sha(Path(path)) == digest for path, digest in pins.items())
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(passed=True, norms=result['norms'],
        common_descent=result['first_order_common_descent_interval'], compositions=compositions)))

if __name__ == '__main__':
    main()
