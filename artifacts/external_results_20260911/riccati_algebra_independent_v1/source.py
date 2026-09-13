"""Independent algebra-only checks; no planner, policy, model or physics imports."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def expanded(A, B, lx, lxx, lu, luu, v, V, k, K):
    qx, qu = lx + A.T @ v, lu + B.T @ v
    qxx = lxx + A.T @ V @ A
    quu, qux = luu + B.T @ V @ B, B.T @ V @ A
    next_v = qx + K.T @ quu @ k + K.T @ qu + qux.T @ k
    next_V = qxx + K.T @ quu @ K + K.T @ qux + qux.T @ K
    return next_v, (next_V + next_V.T) * .5


def factored(A, B, lx, lxx, lu, luu, v, V, k, K):
    F = A + B @ K
    next_v = lx + K.T @ (lu + luu @ k) + F.T @ (v + V @ B @ k)
    next_V = lxx + K.T @ luu @ K + F.T @ V @ F
    return next_v, (next_V + next_V.T) * .5


def main(args):
    assert np.finfo(np.longdouble).eps < np.finfo(np.float64).eps
    rng = np.random.default_rng(773)
    nx, nu = 58, 23
    rows = []
    stored = {}
    for name in ('ordinary', 'zero_feedback', 'large_feedback', 'near_cancelling'):
        A, B = rng.normal(size=(nx, nx)) * .1, rng.normal(size=(nx, nu)) * .1
        J, L = rng.normal(size=(nx, nx)), rng.normal(size=(nx, nx))
        V, lxx = J.T @ J, L.T @ L * .01
        lx, lu, v, k = [rng.normal(size=n) for n in (nx, nu, nx, nu)]
        luu = np.eye(nu) * .0002
        K = rng.normal(size=(nu, nx))
        if name == 'zero_feedback':
            K[:] = 0
        if name in ('large_feedback', 'near_cancelling'):
            K *= 1e4
            V *= 1e4
        if name == 'near_cancelling':
            A = -B @ K + rng.normal(size=A.shape) * 1e-3
        arrays = (A, B, lx, lxx, lu, luu, v, V, k, K)
        old = expanded(*arrays)
        new = factored(*arrays)
        truth = factored(*(a.astype(np.longdouble) for a in arrays))
        truth_expanded = expanded(*(a.astype(np.longdouble) for a in arrays))
        errors = {}
        for index, quantity in enumerate(('gradient', 'hessian')):
            scale = float(np.max(np.abs(truth[index])))
            errors[quantity] = dict(
                truth_scale=scale,
                expanded_max_absolute_error=float(np.max(np.abs(old[index] - truth[index]))),
                factored_max_absolute_error=float(np.max(np.abs(new[index] - truth[index]))),
                extended_precision_formula_difference=float(np.max(np.abs(truth[index] - truth_expanded[index]))),
                factored_relative_error=float(np.max(np.abs(new[index] - truth[index]))) / max(scale, 1e-300),
            )
            # Forming F still subtracts A and BK in this artificial stress case.
            # Its input cancellation cannot be removed by rearranging the value update.
            relative_bound = 2e-7 if name == 'near_cancelling' else 2e-12
            errors[quantity]['declared_relative_check_bound'] = relative_bound
            assert errors[quantity]['factored_relative_error'] < relative_bound
            if name in ('ordinary', 'zero_feedback'):
                np.testing.assert_allclose(old[index], new[index], rtol=2e-12, atol=2e-12)
        rows.append(dict(name=name, errors=errors,
            expanded_hessian_min_eigenvalue=float(np.linalg.eigvalsh(old[1]).min()),
            factored_hessian_min_eigenvalue=float(np.linalg.eigvalsh(new[1]).min())))
        for label, value in zip(('A', 'B', 'lx', 'lxx', 'lu', 'luu', 'v', 'V', 'k', 'K'), arrays):
            stored[name + '_' + label] = value
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / 'matrices.npz', **stored)
    report = dict(kind='independent_synthetic_riccati_algebra_check',
        no_physics_or_controller_execution=True,
        assumption='zero mixed state-control stage cost, as in current tracker expansion',
        variables='V,v are the unmodified next-knot value Hessian and gradient',
        seed=773, nx=nx, nu=nu, float64_epsilon=float(np.finfo(np.float64).eps),
        extended_epsilon=float(np.finfo(np.longdouble).eps), cases=rows,
        no_real_failure_or_policy_qualification=True,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.output / 'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
