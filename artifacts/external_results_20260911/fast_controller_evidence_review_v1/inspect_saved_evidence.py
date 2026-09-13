"""Read existing artifacts only; no inference, fitting, or physical simulation."""
import hashlib
import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
REPO = Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
STUDENT = Path('E:/codex_sonic_runtime/mpc_student_20260910')
OLD = Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

relative = [
    'teacher_3s_perturbed_v1/report.json',
    'teacher_3s_perturbed_v1/expert_samples.npz',
    'student_3s_fit_v1/request.json',
    'student_3s_fit_v1/metrics.json',
    'student_3s_fit_v1/student_snapshot.py',
    'student_3s_fit_v1/trainer_snapshot.py',
    'student_3s_fit_v1/replay_nominal/report.json',
    'student_3s_fit_v1/replay_case8/report.json',
    'bfm_residual_labels_3s_v1/report.json',
    'bfm_residual_labels_3s_v1/labels.npz',
    'bfm_residual_fit_3s_r025_v1/request.json',
    'bfm_residual_fit_3s_r025_v1/metrics.json',
    'bfm_residual_fit_3s_r025_v1/student_snapshot.py',
    'bfm_residual_fit_3s_r025_v1/trainer_snapshot.py',
    'bfm_residual_fit_3s_r025_v1/replay_zero/report.json',
    'bfm_residual_fit_3s_r025_v1/replay_nominal/report.json',
    'bfm_residual_fit_3s_r025_v1/replay_case8/report.json',
    'teacher_native323_full_perturbed_v1/report.json',
    'teacher_native323_full_perturbed_v1/source_phase_metrics_v2.json',
    'walk003_fixedcap_capacity_v1/report.json',
    'walk003_fixedcap_capacity_v1/arrays.npz',
]
files = [STUDENT / r for r in relative]
files += [REPO / 'artifacts/teleop_six_hour_20260910/residual_milestones_v1/residual_00800/summary.json',
          REPO / 'artifacts/g1_true23_bfm_residual_20260911_v1/train3h_v1/request.json',
          REPO / 'artifacts/g1_true23_bfm_residual_20260911_v1/train3h_v1/outcome.json',
          OLD / 'bfm_online_intent_v2/walk003_terminal_bfm_yaw4_hybrid_v1/report.json',
          OLD / 'bfm_online_intent_v2/walk003_terminal_bfm_yaw4_hybrid_v1/recorded_source_audit_v2.json']
inventory = []
for p in files:
    row = dict(path=str(p), exists=p.exists())
    if p.exists():
        row.update(bytes=p.stat().st_size, sha256=sha(p))
        if p.suffix == '.npz':
            with np.load(p, allow_pickle=False) as arrays:
                row['arrays'] = {k: dict(shape=list(arrays[k].shape), dtype=str(arrays[k].dtype)) for k in arrays.files}
    inventory.append(row)
(OUT / 'evidence_inventory.json').write_text(json.dumps(inventory, indent=2), encoding='utf-8')

with np.load(STUDENT / 'walk003_fixedcap_capacity_v1/arrays.npz') as a:
    source = slice(350, 1169)
    residual = a['residual_target'][source]
    error = a['oracle_applied_target'][source] - a['expert_target'][source]
    capacity = dict(source_controls=len(residual),
                    fraction_components_outside_cap=float(np.mean(np.abs(residual) > .25)),
                    fraction_controls_outside_cap=float(np.mean(np.any(np.abs(residual) > .25, axis=1))),
                    absolute_residual_p95=float(np.percentile(np.abs(residual), 95)),
                    absolute_residual_max=float(np.max(np.abs(residual))),
                    applied_oracle_rmse=float(np.sqrt(np.mean(error**2))),
                    applied_oracle_legs_rmse=float(np.sqrt(np.mean(error[:, :12]**2))),
                    applied_oracle_arms_rmse=float(np.sqrt(np.mean(error[:, 13:]**2))),
                    applied_oracle_max=float(np.max(np.abs(error))))
    assert np.array_equal(a['source_frame'][source], np.arange(361, 1180))
report = json.loads((STUDENT / 'walk003_fixedcap_capacity_v1/report.json').read_text())
saved = report['metrics']['full_source_motion']
for ours, theirs in [('applied_oracle_rmse', 'best_possible_applied_target_rmse_rad'),
                    ('applied_oracle_legs_rmse', 'best_possible_leg_target_rmse_rad'),
                    ('applied_oracle_arms_rmse', 'best_possible_arm_target_rmse_rad'),
                    ('applied_oracle_max', 'best_possible_applied_target_max_rad'),
                    ('fraction_components_outside_cap', 'fraction_components_outside_fixed_cap'),
                    ('fraction_controls_outside_cap', 'fraction_controls_with_any_outside_cap')]:
    assert capacity[ours] == saved[theirs]
with np.load(STUDENT / 'bfm_residual_labels_3s_v1/labels.npz') as a:
    r = a['residual_target']
    short = dict(fraction_components_outside_cap=float(np.mean(np.abs(r) > .25)),
                 residual_label_oracle_rmse=float(np.sqrt(np.mean((np.clip(r, -.25, .25) - r)**2))))
result = dict(kind='read_only_saved_array_crosschecks',
              full_walk003_source_capacity=capacity, short_teacher_residual_capacity=short,
              full_source_frame_sequence_exact=True,
              report_values_recomputed_exact=True,
              new_inference=False, new_training=False, new_physics=False)
(OUT / 'independent_array_crosschecks.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
