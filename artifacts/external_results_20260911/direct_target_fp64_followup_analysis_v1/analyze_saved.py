"""Pure saved-array objective analysis. No model, optimizer, or physics imports."""
from pathlib import Path
import hashlib
import json
import numpy as np

BASE = Path(__file__).resolve().parent
NEW = BASE.parent

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def stats(a):
    a = np.asarray(a, dtype=np.float64)
    return dict(min=float(a.min()), p50=float(np.median(a)), p95=float(np.percentile(a, 95)),
                max=float(a.max()), rms=float(np.sqrt(np.mean(a * a))))

def main():
    paths = dict(
        centers=NEW/'velocity_chord_student_v1/generation/centers.npz',
        teacher=NEW/'velocity_chord_student_v1/generation/teacher_target.npy',
        feedback_clip=NEW/'velocity_chord_student_v1/generation/teacher_feedback_clipped.npy',
        native_clip=NEW/'velocity_chord_student_v1/generation/teacher_native_clipped.npy',
        branch=NEW/'velocity_chord_student_v1/generation/teacher_branch_changed.npy',
        generated=NEW/'velocity_chord_student_v1/generation/report.json',
        old_nominal=NEW/'direct_target_student_v1/fit/final_GPU_nominal.npy',
        old_velocity=NEW/'direct_target_student_v1/fit/final_GPU_velocity.npy',
        old_metrics=NEW/'direct_target_student_v1/fit/final_metrics.json',
        new_nominal=NEW/'direct_target_fp64_export_v1/export/CPU64_nominal.npy',
        new_velocity=NEW/'direct_target_fp64_export_v1/export/CPU64_velocity.npy',
        new_metrics=NEW/'direct_target_fp64_export_v1/export/metrics_CPU64.json',
        generation_review=NEW/'velocity_chord_completed_data_review_v1/review.json',
        comparison=NEW/'direct_target_continuation_saved_diagnostics_v1/report.json',
        objective=NEW/'direct_target_continuation_v1/source_snapshot_v1/direct_objective.py',
        norm=NEW/'direct_target_continuation_v1/fit/normalization.npz',
        native_plans=NEW/'bfm_entry250_actual_oracle_v1/nominal/plans.json',
        expert_outcome=NEW/'bfm_entry250_actual_oracle_v1/outcome.json',
        actual_report=NEW/'direct_target_fp64_export_evaluation_v2/nominal/report.json',
        actual_trace=NEW/'direct_target_fp64_export_evaluation_v2/nominal/trace.npz',
        actual_owner=NEW/'direct_target_fp64_export_evaluation_v2/evaluation_completion_verification.json',
    )
    pins = {str(p): sha(p) for p in paths.values()}
    request = dict(kind='pure_saved_secant_scale_and_phase_analysis', input_sha256=pins,
                   source_sha256=sha(__file__), model_calls=0, optimizer_calls=0, native_calls=0)
    if (BASE/'request.json').exists():
        raise FileExistsError('Preserve existing analysis; no automatic overwrite.')
    write(BASE/'request.json', request)
    z = np.load(paths['centers'], allow_pickle=False)
    span = z['joint_span'].astype(np.float64)
    teacher = np.load(paths['teacher'], allow_pickle=False)
    change = (teacher - z['expert_target'][:, None, None, :]) / span
    signed_radius = np.asarray([-0.01, 0.01], dtype=np.float64)
    teacher_secant = change / signed_radius[None, None, :, None]
    cells = [np.flatnonzero((z['dataset'] == d) & phase) for d in range(3)
             for phase in (z['control'] < 350, (z['control'] >= 350) & (z['control'] < 1169), z['control'] >= 1169)]
    assert list(map(len, cells)) == [100, 819, 100] * 3
    assert np.array_equal(z['dataset'], np.repeat(np.arange(3), 1019))
    assert np.array_equal(z['control'], np.tile(np.arange(250, 1269), 3))
    eqmean = lambda a: float(np.mean([np.mean(a[ids]) for ids in cells]))
    feedback = np.load(paths['feedback_clip'], allow_pickle=False)
    native = np.load(paths['native_clip'], allow_pickle=False)
    branch = np.load(paths['branch'], allow_pickle=False)
    feedback_probe = np.any(feedback, axis=-1)
    native_probe = np.any(native, axis=-1)
    branch_probe = np.any(branch, axis=-1) if branch.ndim == 4 else branch
    masks = dict(all=np.ones(change.shape[:3], dtype=bool),
                 feedback_clip=feedback_probe, native_clip=native_probe,
                 any_clip=feedback_probe | native_probe, unclipped=~(feedback_probe | native_probe),
                 branch_changed=branch_probe)
    results = {}
    for label in ('old', 'new'):
        nominal = np.load(paths[label+'_nominal'], allow_pickle=False).astype(np.float64)[:3057]
        probe = np.load(paths[label+'_velocity'], allow_pickle=False).astype(np.float64).reshape(3057, 23, 2, 23)
        student_change = probe - nominal[:, None, None, :]
        error = student_change - change
        # Odd term is symmetric finite slope; even term exposes curvature or bias.
        odd_error = (error[:, :, 1] - error[:, :, 0]) / 2
        even_error = (error[:, :, 1] + error[:, :, 0]) / 2
        losses = read(paths[label+'_metrics'])
        reconstructed = eqmean(error ** 2)
        assert np.isclose(reconstructed, losses['full_velocity_objective'], rtol=1e-12, atol=0)
        assert np.isclose(eqmean(odd_error**2) + eqmean(even_error**2), reconstructed, rtol=1e-12, atol=0)
        nominal_loss = losses['nominal_objective']
        physical_loss = losses['physical_objective']
        groups = {}
        for name, mask in masks.items():
            groups[name] = dict(probe_rows=int(mask.sum()),
                raw_count_weighted_chord_MSE=float(np.mean(error[mask]**2)),
                raw_count_weighted_signed_secant_MSE=float(np.mean((error[mask] / .01)**2)),
                raw_count_weighted_teacher_secant_RMS=float(np.sqrt(np.mean(teacher_secant[mask]**2))))
        results[label] = dict(
            nominal_loss=nominal_loss, velocity_loss=reconstructed, physical_loss=physical_loss,
            original_scalar_sum=nominal_loss+reconstructed+physical_loss,
            velocity_scalar_share=reconstructed/(nominal_loss+reconstructed+physical_loss),
            signed_secant_loss=eqmean((error/signed_radius[None,None,:,None])**2),
            secant_scalar_to_nominal_ratio=10000*reconstructed/nominal_loss,
            secant_scalar_share_if_coefficient_one=10000*reconstructed/(nominal_loss+10000*reconstructed+physical_loss),
            odd_error_MSE=eqmean(odd_error**2), even_error_MSE=eqmean(even_error**2),
            teacher_chord_MSE=eqmean(change**2), teacher_signed_secant_MSE=eqmean(teacher_secant**2),
            student_signed_secant_MSE=eqmean((student_change/signed_radius[None,None,:,None])**2),
            per_cell=[dict(dataset=i//3, phase=i%3, rows=len(ids),
                chord_MSE=float(np.mean(error[ids]**2)), signed_secant_MSE=float(np.mean((error[ids]/.01)**2)),
                teacher_secant_MSE=float(np.mean(teacher_secant[ids]**2)),
                nominal_loss=losses['nominal_cells'][i]['normalized_MSE']) for i,ids in enumerate(cells)],
            groups=groups)
    plans = read(paths['native_plans'])
    fresh_ms = []
    for plan in plans:
        for candidate in plan['selection']['candidates']:
            if candidate['candidate'] == 'fresh_BFM_proposal':
                fresh_ms.append(candidate['diagnostics']['elapsed_ms'])
    actual = read(paths['actual_report'])
    assert actual['trace_sha256'] == sha(paths['actual_trace'])
    report = dict(passed=True, request_sha256=sha(BASE/'request.json'), model_calls=0, native_calls=0, optimizer_calls=0,
        formula='signed endpoint-minus-center, divided by signed normalized-native-speed radius 0.01; squared multiplier10000',
        derivative_scope='finite secants through both exact committed-plan clips; not analytic K or a replanned expert',
        objective_results=results,
        teacher_chord_abs=stats(np.abs(change)), teacher_secant_abs=stats(np.abs(teacher_secant)),
        generation_clip_counts={k:read(paths['generated'])[k] for k in ('feedback_clipped_probes','native_clipped_probes','branch_changed_probes','zero_gain_centers')},
        nominal_mass=dict(acquisition=1/3, source=1/3, return_phase=1/3, walk003_three_datasets=3/5,
                          each_query250_acquisition_row=1/1500, each_query250_source_row=1/(15*819),
                          each_pico_source_row=1/(15*5780)),
        expert_plans=len(plans), fresh_seed_ms=stats(fresh_ms),
        expert_nominal_wall_seconds=read(paths['expert_outcome'])['nominal']['elapsed_seconds'],
        actual_outcome={k:actual[k] for k in ('physics_steps','attempted_controls','failure','policy_ms_p50_p95_max','source_metrics')},
        parameter_gradient_claim=False,
        all_inputs_unchanged=all(sha(p)==h for p,h in pins.items()))
    if not report['all_inputs_unchanged']: raise ValueError('Changed immutable input.')
    write(BASE/'saved_scale_report.json', report)
    print(json.dumps(dict(passed=True, report_sha256=sha(BASE/'saved_scale_report.json'),
                         old=results['old']['signed_secant_loss'], new=results['new']['signed_secant_loss'],
                         teacher=results['new']['teacher_signed_secant_MSE'], fresh_seed_ms=report['fresh_seed_ms'])))

if __name__ == '__main__': main()
