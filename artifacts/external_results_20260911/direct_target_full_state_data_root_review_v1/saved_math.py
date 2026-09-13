"""Independent saved-state checks. No engine, inference, optimizer or generator import."""
import ast
from pathlib import Path
from types import SimpleNamespace
import hashlib
import numpy as np

GROUPS = ((0, 3), (3, 6), (6, 29), (29, 32), (32, 35), (35, 58))
KEPT = np.r_[0:52, 75:1023]


def validate_producer_report(report, request_sha256):
    if report['complete'] is not True or report['request_sha256'] != request_sha256:
        raise AssertionError('Complete producer request identity')
    for key, expected in [('centers', 3057), ('signed_rows', 354612),
                          ('overlap_rows', 140622), ('new_rows', 213990),
                          ('pure_feature_calls', 357669), ('pure_committed_map_calls', 357669)]:
        if report[key] != expected:
            raise AssertionError('Producer count: ' + key)
    for key in ('feature_attempted', 'feature_returned', 'map_attempted', 'map_returned'):
        if report['call_accounting'][key] != 357669:
            raise AssertionError('Producer call accounting: ' + key)
    if 'centers.npz' not in report['output_sha256']:
        raise AssertionError('Saved centers output binding')

def original_difference(core_path):
    tree = ast.parse(Path(core_path).read_text())
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name in ('quat_mul', 'quat_log')]
    planner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Planner')
    definitions.append(next(node for node in planner.body if isinstance(node, ast.FunctionDef) and node.name == 'difference'))
    namespace = {'np': np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[])), str(core_path), 'exec'), namespace)
    topology = SimpleNamespace(nq=30, nv=29, lin_q=np.r_[0:3, 7:30], lin_v=np.r_[0:3, 6:29], free=[(0, 0)])
    return lambda plan, actual: namespace['difference'](topology, plan[None], actual[None])[0]

def exact(actual, expected, name):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or actual.dtype != expected.dtype or actual.tobytes() != expected.tobytes():
        raise AssertionError(name)

def expected_radii(caps):
    values = np.empty(58, np.float64)
    for indices, value in [(slice(0, 3), .001), (slice(3, 6), .01),
                           (slice(6, 29), .01), (slice(29, 32), .05), (slice(32, 35), .25)]:
        values[indices] = value
    values[35:] = np.asarray(caps, np.float64) * .01
    return values

def check_probe(q0, v0, q, v, axis, displacement, plan, difference, limits, caps):
    if q.dtype != np.float64 or v.dtype != np.float64 or q.shape != (30,) or v.shape != (29,):
        raise AssertionError('saved state dtype/shape')
    if not np.isfinite(q).all() or not np.isfinite(v).all():
        raise AssertionError('finite state')
    if np.any(q[7:] < limits[:, 0]) or np.any(q[7:] > limits[:, 1]):
        raise AssertionError('native position bound')
    if np.any(np.abs(v[6:]) >= caps) or q[2] <= 0 or abs(np.linalg.norm(q[3:7]) - 1) > 1e-10:
        raise AssertionError('speed/height/quaternion bound')
    expected_q, expected_v = q0.copy(), v0.copy()
    if axis < 3:
        expected_q[axis] += displacement
    elif axis < 6:
        # Rotation is checked in the same plan chart, independently of its construction.
        expected_q[3:7] = q[3:7]
    elif axis < 29:
        expected_q[axis + 1] += displacement
    else:
        expected_v[axis - 29] += displacement
    exact(q, expected_q, 'only requested position coordinate changes')
    exact(v, expected_v, 'only requested velocity coordinate changes')
    tangent = difference(plan, np.r_[q, v])
    change = tangent - difference(plan, np.r_[q0, v0])
    wanted = np.zeros(58, np.float64)
    wanted[axis] = displacement
    if not np.allclose(change, wanted, rtol=0, atol=2e-12):
        raise AssertionError('single plan-chart axis')
    if axis in (3, 4, 5) and np.linalg.norm(tangent[3:6]) >= np.pi - 1e-6:
        raise AssertionError('rotation chart boundary')
    return tangent, change

def map_values(gain, tangent, plan_target, limits):
    raw = gain @ tangent
    clipped = np.clip(raw, -.1, .1)
    preclip = plan_target + clipped
    target = np.clip(preclip, limits[:, 0], limits[:, 1])
    return raw, clipped, preclip, target

def canonical_feature_digest(feature):
    # IEEE negative zero has the same numerical network input as positive zero.
    row = np.asarray(feature).copy()
    if row.dtype != np.float32 or row.shape != (1000,) or not np.isfinite(row).all():
        raise AssertionError('finite float32 feature row')
    row[row == 0] = 0
    return hashlib.sha256(row.tobytes()).digest()

def duplicate_conflicts(features, targets, spans, defaults, index, corpus):
    """Return exact collisions for one streamed corpus; never silently remove rows."""
    spans, defaults = np.asarray(spans), np.asarray(defaults)
    if spans.shape != (23,) or defaults.shape != (23,) or not np.isfinite(spans).all() or not np.isfinite(defaults).all() or np.any(spans <= 0):
        raise AssertionError('Finite positive normalization span/default')
    if np.shape(targets) != (len(features), 23):
        raise AssertionError('Target corpus shape')
    duplicates, conflicts = [], []
    for row in range(len(features)):
        feature = np.asarray(features[row])
        key = canonical_feature_digest(feature)
        target = np.asarray(targets[row])
        if not np.isfinite(target).all():
            raise AssertionError('Finite target row')
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            label = ((target - defaults) / spans).astype(np.float32)
        if not np.isfinite(label).all():
            raise AssertionError('Finite normalized target row')
        if key in index:
            old_feature, old_label, old_id, old_target = index[key]
            if not np.array_equal(feature, old_feature):
                raise AssertionError('feature hash collision')
            entry = dict(first=old_id, repeated=[corpus, row],
                         raw_target_max_difference=float(np.max(np.abs(np.asarray(targets[row])-old_target))),
                         normalized_label_max_difference=float(np.max(np.abs(label.astype(np.float64)-old_label.astype(np.float64)))))
            duplicates.append(entry)
            if not np.array_equal(label, old_label):
                conflicts.append(entry)
        else:
            # Callers using large corpora should retain indexed views, not full copies.
            index[key] = (feature, label, [corpus, row], np.asarray(targets[row]))
    return duplicates, conflicts
