"""Independently recompute the two archived real Riccati cancellation witnesses."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from verify_riccati_closed_loop_algebra import expanded, factored


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    assert np.finfo(np.longdouble).eps < np.finfo(np.float64).eps
    paths = [args.input / name for name in (
        'guided_initial_expanded_mu1e+00_knot13.npz',
        'refined_final_expanded_mu1e+05_knot13.npz',
    )]
    result = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            a = {key: archive[key].copy() for key in archive.files}
        parameters = [a[key] for key in (
            'A', 'B', 'lx', 'lxx', 'lu', 'luu', 'next_gradient', 'next_value', 'k', 'K')]
        old, new = expanded(*parameters), factored(*parameters)
        np.testing.assert_array_equal(old[0], a['expanded_x'])
        np.testing.assert_array_equal(old[1], a['expanded_xx'])
        # Equivalent multiplication parenthesizations can round differently.
        truth = factored(*(value.astype(np.longdouble) for value in parameters))
        row = dict(input=str(path), input_sha256=sha(path), archived_expanded_reproduced=True,
                   native_joint_limits_changed=False, physical_steps=0, entries={})
        for i, quantity in enumerate(('gradient', 'hessian')):
            scale = float(np.max(np.abs(truth[i])))
            row['entries'][quantity] = dict(
                extended_truth_scale=scale,
                expanded_max_absolute_error=float(np.max(np.abs(old[i] - truth[i]))),
                factored_max_absolute_error=float(np.max(np.abs(new[i] - truth[i]))),
                expanded_relative_error=float(np.max(np.abs(old[i] - truth[i]))) / max(scale, 1e-300),
                factored_relative_error=float(np.max(np.abs(new[i] - truth[i]))) / max(scale, 1e-300),
                archived_grouped_max_difference=float(np.max(np.abs(new[i] - a['grouped_x' if i == 0 else 'grouped_xx']))),
            )
        row['minimum_hessian_eigenvalues'] = dict(
            expanded=float(np.linalg.eigvalsh(old[1]).min()),
            factored=float(np.linalg.eigvalsh(new[1]).min()),
            extended_then_cast=float(np.linalg.eigvalsh(truth[1].astype(np.float64)).min()),
        )
        assert row['entries']['hessian']['factored_max_absolute_error'] < row['entries']['hessian']['expanded_max_absolute_error']
        result.append(row)
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(kind='independent_real_saved_matrix_arithmetic_only', witnesses=result,
        full_backward_or_controller_qualified=False,
        limitation='Factorization reduces cancellation; it cannot restore a diagonal smaller than floating-point resolution in a later normal equation.',
        source_hashes={str(p): sha(p) for p in (
            Path(__file__), Path(__file__).with_name('verify_riccati_closed_loop_algebra.py'))})
    for path in paths:
        (args.output / path.name).write_bytes(path.read_bytes())
    (args.output / 'source.py').write_bytes(Path(__file__).read_bytes())
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    main(parser.parse_args())
