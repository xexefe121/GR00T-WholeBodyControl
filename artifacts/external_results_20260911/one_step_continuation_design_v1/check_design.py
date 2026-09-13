"""Saved inputs and synthetic algebra only; no network/physics/label construction."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

BASE = Path(__file__).resolve().parent
NEW = BASE.parent
V = NEW / 'velocity_chord_student_v1'
SOURCE = V / 'source_snapshot_v3'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def array_sha(a):
    a = np.asarray(a)
    return dict(shape=list(a.shape), dtype=str(a.dtype), sha256=hashlib.sha256(a.tobytes()).hexdigest())

paths = [BASE / 'DESIGN.md', Path(__file__), V / 'generation/centers.npz',
         V / 'fit/student_head.pt', V / 'fit/student_head.onnx', V / 'fit/report.json',
         V / 'fit/request.json', V / 'fit/final_predictions.npz',
         V / 'fit/final_chord_metrics.json', V / 'fit/final_nominal_metrics.json',
         V / 'training_runtime_identity.json',
         NEW / 'velocity_chord_final_export_review_v1/review.json',
         NEW / 'velocity_fit_evidence_independent_v1/report.json',
         NEW / 'old_expert_prefix_snapshots_v2/completion_verification.json',
         NEW / 'old_expert_prefix_snapshots_v2/capture/report.json',
         NEW / 'old_expert_prefix_snapshots_v2/capture/control_snapshots.npz',
         NEW / 'previous_command_perturbation_feasibility_v1/behavior_report.json']
paths += [SOURCE / name for name in ('fit_velocity_chords.py', 'chord_training_objective.py',
    'fit_phase_fullbatch_once.py', 'fit_linear_head.py', 'continue_linear_fit.py',
    'chord_fit_diagnostics.py', 'chord_head_sensitivity.py', 'student_linear_runtime.py')]
pins = {p.as_posix(): sha(p) for p in paths}
assert not (BASE / 'report.json').exists()
checks = []
def check(name, passed):
    checks.append(dict(name=name, passed=bool(passed)))
    assert passed, name

with np.load(V / 'generation/centers.npz', allow_pickle=False) as z:
    control = z['control'].copy()
    dataset = z['dataset'].copy()
    frame = z['source_frame'].copy()
check('all3057 canonical center identities',
      np.array_equal(control, np.tile(np.arange(250, 1269), 3)) and
      np.array_equal(dataset, np.repeat(np.arange(3), 1019)))
check('all3057 original frames', np.array_equal(frame, control + 11))
rows = []
cells = []
for d, name in enumerate(('old', 'query1', 'query250')):
    for c in range(250, 1268):
        start = d * 1019 + c - 250
        end = start + 1
        rows.append((d, c, c + 1, start, end))
        check(f'pair {d}/{c}: same dataset successor/frame',
              dataset[start] == dataset[end] == d and control[end] == c + 1
              and frame[end] == c + 12)
    for phase, lo, hi, count, nominal in (
        ('acquisition', 251, 349, 99, 100), ('source', 350, 1168, 819, 819),
        ('return', 1169, 1268, 100, 100)):
        indices = [r for r in rows if r[0] == d and lo <= r[2] <= hi]
        check(f'{name}/{phase} requested coverage', len(indices) == count)
        cells.append(dict(dataset=name, phase=phase, nominal_rows=nominal,
            requested_physical_rows=count, successor_range=[lo, hi], nominal_cell_weight='1/9',
            physical_cell_weight='1/9', physical_row_joint_weight=f'1/(9*{count}*23)',
            valid_rows='not produced/bound at design time', empty_cell='zero, no redistribution'))
check('exact3054 starts and matching successors', len(rows) == 3054)
check('boundary crossings preserved', all(any(r[0] == d and r[1] == c for r in rows)
      for d in range(3) for c in (349, 1168, 1267)))

# Fixed small arithmetic examples are not robot labels or head evaluations.
s = np.array([2., 4.], np.float64)
bn, bp = np.array([1., -2.]), np.array([3., 1.])
tn, tp = np.array([1.5, -1.]), np.array([2., 2.])
fn, fp = np.array([.2, -.1]), np.array([-.3, .4])
en, ep = (bn + s * fn - tn) / s, (bp + s * fp - tp) / s
pair = ((bp + s * fp) - (bn + s * fn) - (tp - tn)) / s
residual_pair = fp - fn - ((tp - bp) - (tn - bn)) / s
check('total proposal response equals error difference', np.allclose(pair, ep - en, atol=1e-15, rtol=0))
check('response residual algebra includes both bases', np.allclose(pair, residual_pair, atol=1e-15, rtol=0))
wrong = fp - fn - (tp - tn) / s
check('omitting base response changes label', not np.allclose(pair, wrong))
check('pair loss distinct from absolute branch MSE', not np.allclose(pair * pair, ep * ep))
same_error = np.array([.03, -.06])
check('equal nonzero center/branch error has zero pair but nonzero anchor loss',
      np.all((same_error - same_error) == 0) and np.sum(same_error ** 2) > 0)
check('fixed failed-row denominator does not boost survivors', 1 / (9 * 99 * 23) < 1 / (9 * 2 * 23))
nan_rows = np.array([[1., 2.], [np.nan, np.nan]])
mask = np.array([True, False])
check('index valid before arithmetic avoids NaN times zero', np.isfinite((nan_rows[mask] ** 2).sum()))

# Deserialization only: no actor, optimizer, RNG restoration, or forward call.
saved = torch.load(V / 'fit/student_head.pt', map_location='cpu', weights_only=True)
check('ordinary70000 checkpoint', saved['completed_steps'] == 70000)
state = saved['optimizer_state']['state']
check('six complete optimizer states at70000', len(state) == 6 and all(int(x['step']) == 70000 for x in state.values()))
groups = saved['optimizer_state']['param_groups']
check('restored last lr and weight decay', all(g['lr'] == 3e-7 and g['weight_decay'] == 1e-5 for g in groups))
for key in ('feature_mean', 'feature_std', 'joint_span'):
    check(f'{key} finite float32', saved[key].dtype == torch.float32 and bool(torch.isfinite(saved[key]).all()))
check('original norm dimensions', saved['feature_mean'].shape == (1069,) and saved['feature_std'].shape == (1069,)
      and saved['joint_span'].shape == (23,))
check('positive old scaling', bool((saved['feature_std'] > 0).all()) and bool((saved['joint_span'] > 0).all()))
fit = json.loads((V / 'fit/report.json').read_text())
check('final checkpoint and ONNX bind passedfit', fit['checkpoint_sha256'] == sha(V / 'fit/student_head.pt')
      and fit['onnx_sha256'] == sha(V / 'fit/student_head.onnx') and fit['numerical_gate_passed'])
check('prior row budget', fit['attempt_counters']['training_head_rows_returned'] == 21045000)
max_rows = 5000 * (3057 + 1152 + 3054)
check('max training cost', max_rows == 36315000)
check('max diagnostic call budget', 1126 + 2 * int(np.ceil(3054 / 256)) == 1150)
check('all frozen inputs remain unchanged', all(sha(p) == h for p, h in pins.items()))
report = dict(kind='read_only_one_step_continuation_objective_design', passed=True,
    selected_fit=False, optimizer_updates=0, network_evaluations=0, inference_calls=0,
    physics_steps=0, new_labels=0, checkpoint_deserialization_only=True,
    source_and_input_sha256=pins, checks=checks, check_count=len(checks),
    nominal_rows=3057, requested_branch_rows=3054, nine_cells=cells,
    proposed_objective='L_nominal + unchanged L_velocity + L_physical_finite_response',
    proposed_updates=[70001, 75000], proposed_additional_updates=5000,
    proposed_lr_reset_explicit=True, ordinary_final_only=True,
    old_scaling={k:array_sha(saved[k].numpy()) for k in ('feature_mean', 'feature_std', 'joint_span')},
    training_rows_max=max_rows, prior_training_rows=21045000,
    ratio=max_rows / 21045000, prior_seconds_parent_estimate=485.,
    seconds_linear_row_scaling_estimate=485 * max_rows / 21045000,
    max_initial_final_torch_diagnostic_rows=2 * (143679 + 7 + 3054),
    max_initial_final_head_onnx_calls=1150,
    limitation='No actual branch-validity mask, new fit or behavioral qualification at design time.')
(BASE / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
print(json.dumps(dict(passed=True, checks=len(checks), report_sha256=sha(BASE / 'report.json'),
    design_sha256=sha(BASE / 'DESIGN.md'), network_evaluations=0, physics_steps=0)))
