"""Pure fixed full-state sampler, schedule and calibration math."""
import math
import numpy as np

GROUP_SIZES = (3, 3, 23, 3, 3, 23)
CELL_COUNTS = (100, 819, 100) * 3
PAIRS_PER_CELL = 16
CELLS = 54
PAIRS = CELLS * PAIRS_PER_CELL
ENDPOINTS = 2 * PAIRS
UPDATES = 10000
SEED = 20260911
START_GLOBAL_STEP = 55000
FINAL_GLOBAL_STEP = 65000

def center_cells(dataset, control, axis_group):
    dataset = np.asarray(dataset)
    control = np.asarray(control)
    axis_group = np.asarray(axis_group)
    if dataset.dtype != np.int64 or control.dtype != np.int64 or axis_group.dtype != np.int8:
        raise ValueError('Fixed integer metadata dtype.')
    if not np.array_equal(dataset, np.repeat(np.arange(3, dtype=np.int64), 1019)):
        raise ValueError('Fixed three datasets/1019 centers each.')
    if not np.array_equal(control, np.tile(np.arange(250, 1269, dtype=np.int64), 3)):
        raise ValueError('Fixed center controls250..1268.')
    if not np.array_equal(axis_group, np.repeat(np.arange(6, dtype=np.int8), GROUP_SIZES)):
        raise ValueError('Fixed six ordered state groups.')
    cells = [np.flatnonzero((dataset == d) & mask) for d in range(3)
             for mask in (control < 350, (control >= 350) & (control < 1169), control >= 1169)]
    if tuple(map(len, cells)) != CELL_COUNTS:
        raise ValueError('Fixed acquisition/source/return counts.')
    return cells

def build_schedule(dataset, control, axis_group, updates=UPDATES, seed=SEED):
    """Generate once before optimization. Smaller updates permitted for synthetic tests."""
    if isinstance(updates, bool) or not isinstance(updates, int) or not 1 <= updates <= UPDATES:
        raise ValueError('Fixed schedule size.')
    cells = center_cells(dataset, control, axis_group)
    axes_by_group = [np.flatnonzero(axis_group == g) for g in range(6)]
    generator = np.random.Generator(np.random.PCG64(seed))
    initial_rng = generator.bit_generator.state
    centers = np.empty((updates, PAIRS), dtype=np.int32)
    axes = np.empty((updates, PAIRS), dtype=np.int8)
    for update in range(updates):
        for cell, eligible_centers in enumerate(cells):
            for group, eligible_axes in enumerate(axes_by_group):
                target = slice((cell * 6 + group) * PAIRS_PER_CELL, (cell * 6 + group + 1) * PAIRS_PER_CELL)
                selected_centers = generator.integers(0, len(eligible_centers), size=PAIRS_PER_CELL, dtype=np.int32)
                selected_axes = generator.integers(0, len(eligible_axes), size=PAIRS_PER_CELL, dtype=np.int32)
                centers[update, target] = eligible_centers[selected_centers]
                axes[update, target] = eligible_axes[selected_axes]
    return centers, axes, initial_rng, generator.bit_generator.state

def verify_schedule(centers, axes, dataset, control, axis_group, updates=UPDATES):
    cells = center_cells(dataset, control, axis_group)
    if centers.shape != (updates, PAIRS) or centers.dtype != np.int32:
        raise ValueError('Fixed center schedule schema.')
    if axes.shape != centers.shape or axes.dtype != np.int8:
        raise ValueError('Fixed axis schedule schema.')
    for cell, eligible_centers in enumerate(cells):
        for group in range(6):
            selected = slice((cell * 6 + group) * PAIRS_PER_CELL, (cell * 6 + group + 1) * PAIRS_PER_CELL)
            if not np.isin(centers[:, selected], eligible_centers).all():
                raise ValueError('Sample outside its dataset/phase cell.')
            if not np.isin(axes[:, selected], np.flatnonzero(axis_group == group)).all():
                raise ValueError('Sample outside its tangent group.')

def endpoint_rows(centers, axes):
    centers = np.asarray(centers)
    axes = np.asarray(axes)
    if centers.shape != (PAIRS,) or axes.shape != centers.shape:
        raise ValueError('One864-pair schedule row required.')
    if np.any(centers < 0) or np.any(centers >= 3057) or np.any(axes < 0) or np.any(axes >= 58):
        raise ValueError('Endpoint index outside fixed corpus.')
    # Widen before arithmetic: stored int8/int32 indices must never overflow.
    return (((centers.astype(np.int64) * 58 + axes.astype(np.int64))[:, None] * 2)
            + np.arange(2, dtype=np.int64)[None]).reshape(ENDPOINTS)

def cosine_rate(index):
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < UPDATES:
        raise ValueError('Update index outside fixed schedule.')
    return 1e-5 + .5 * (1e-4 - 1e-5) * (1 + math.cos(math.pi * index / (UPDATES - 1)))

def calibrate(gradients):
    """One initial Euclidean norm balance; no dynamic or per-group coefficient."""
    names = ('nominal', 'full_state', 'physical')
    vectors = {}
    expected_shapes = None
    for name in names:
        pieces = gradients[name]
        if len(pieces) != 6:
            raise ValueError('Six ordered parameter gradients required.')
        shapes = [np.asarray(piece).shape for piece in pieces]
        if expected_shapes is None:
            expected_shapes = shapes
        elif shapes != expected_shapes:
            raise ValueError('Ordered parameter-gradient shapes differ.')
        vectors[name] = np.concatenate([np.asarray(piece, dtype=np.float64).reshape(-1) for piece in pieces])
        if not np.isfinite(vectors[name]).all():
            raise ValueError('Nonfinite returned parameter gradient.')
    if len({len(vector) for vector in vectors.values()}) != 1:
        raise ValueError('Parameter-gradient dimensions differ.')
    norms = {name: float(np.linalg.norm(vector)) for name, vector in vectors.items()}
    if any(not np.isfinite(norms[name]) or norms[name] <= 0 for name in ('nominal', 'full_state')):
        raise ValueError('Cannot calibrate with zero/nonfinite nominal or full-state norm.')
    coefficient = norms['nominal'] / norms['full_state']
    if not np.isfinite(coefficient) or coefficient <= 0:
        raise ValueError('Invalid fixed coefficient.')
    combined = vectors['nominal'] + coefficient * vectors['full_state'] + vectors['physical']
    gram = [[float(np.dot(vectors[left], vectors[right])) for right in names] for left in names]
    cosines = [[gram[i][j] / (norms[left]*norms[right]) if norms[left]*norms[right]>0 else None
                for j,right in enumerate(names)] for i,left in enumerate(names)]
    return dict(coefficient=coefficient, names=list(names), norms=norms, gram_matrix=gram,
        cosine_matrix=cosines,
        first_order_descent_dots={name: float(np.dot(vector, combined)) for name, vector in vectors.items()},
        total_gradient_norm=float(np.linalg.norm(combined)),
        weight_sweep=False, dynamic_recalibration=False, AdamW_or_finite_step_guarantee=False)
