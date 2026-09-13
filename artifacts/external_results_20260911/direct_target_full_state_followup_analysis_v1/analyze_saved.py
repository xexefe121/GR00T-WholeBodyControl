"""Bounded saved-only curve/metadata analysis; no model or dynamics imports."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
FIT = ROOT / 'direct_target_full_state_student_v1/fit'
PATHS = {
    'progress': FIT / 'training_progress.npy',
    'nominal_curves': FIT / 'nominal_cell_losses.npy',
    'response_curves': FIT / 'full_state_cell_losses.npy',
    'physical_curves': FIT / 'physical_cell_losses.npy',
    'initial_metrics': FIT / 'initial_GPU32_metrics.json',
    'final_metrics': FIT / 'final_GPU32_metrics.json',
    'coefficient': FIT / 'coefficient.json',
    'response_baseline': ROOT / 'full_state_response_baseline_v1/report.json',
    'prior_hold_baseline': ROOT / 'full_state_response_baseline_v1/previous_target_report.json',
    'walk003_centers': ROOT / 'velocity_chord_student_v1/generation/centers.npz',
    'pico_labels': ROOT / 'pico_walk002_labels_v1/collection/pico/labels.npz',
    'walk002_labels': ROOT / 'pico_walk002_labels_v1/collection/walk002/labels.npz',
    'physical_manifest': ROOT / 'one_step_policy_branch_collection_resume2969_v1/collection/data/manifest.json',
    'history_source': ROOT / 'direct_target_full_state_evaluation_v1/source_draft_v1/gear_sonic/utils/g1_true23_bfm_seed_observations.py',
    'outcome_owner': ROOT / 'direct_target_full_state_evaluation_v1/evaluation_completion_verification.json',
    'outcome_report': ROOT / 'direct_target_full_state_evaluation_v1/nominal/report.json',
    'semantics': ROOT / 'direct_target_full_state_saved_semantics_review_v1/results_v1/report.json',
}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')

def main():
    if sys.argv[1:] == ['--freeze']:
        write_new(BASE / 'request.json', dict(scope='saved curves and causal context availability only',
            input_sha256={p.as_posix(): sha(p) for p in PATHS.values()}, source_sha256=sha(__file__),
            model_calls=0, gradient_calls=0, optimizer_updates=0, native_steps=0))
        return
    assert len(sys.argv) == 1
    request = read(BASE / 'request.json')
    assert request['source_sha256'] == sha(__file__)
    for path, digest in request['input_sha256'].items():
        assert sha(path) == digest, path
    p = np.load(PATHS['progress'], allow_pickle=False)
    assert p.shape == (10000, 6) and p.dtype == np.float64 and np.isfinite(p).all()
    windows = [(0, 100), (0, 1000), (1000, 2000), (4000, 5000), (8000, 9000), (9000, 10000), (9900, 10000)]
    curves = {name: np.load(PATHS[name], allow_pickle=False) for name in
              ('nominal_curves', 'response_curves', 'physical_curves')}
    for name, count in [('nominal_curves', 15), ('response_curves', 54), ('physical_curves', 9)]:
        assert curves[name].shape == (10000, count) and np.isfinite(curves[name]).all()
    initial, final = read(PATHS['initial_metrics']), read(PATHS['final_metrics'])
    objective = {}
    for key in ('nominal_objective', 'full_state_objective', 'physical_objective', 'weighted_objective'):
        objective[key] = dict(initial=initial[key], final=final[key], fraction_change=final[key] / initial[key] - 1)
    per_cell = {}
    for key, metric in [('nominal_cells', 'normalized_MSE'), ('full_state_cells', 'response_MSE'), ('physical_cells', 'response_MSE')]:
        a = np.array([row[metric] for row in initial[key]])
        z = np.array([row[metric] for row in final[key]])
        per_cell[key] = dict(improved=int(np.count_nonzero(z < a)), worsened=int(np.count_nonzero(z > a)),
            fraction_change=(z / a - 1).tolist())
    baseline = read(PATHS['response_baseline'])
    prior = read(PATHS['prior_hold_baseline'])
    context = {}
    for key, expected in [('walk003_centers',3057), ('pico_labels',5980), ('walk002_labels',867)]:
        with np.load(PATHS[key], allow_pickle=False) as z:
            assert z['previous_action'].shape == (expected,23) and z['previous_action'].dtype == np.float32
            assert z['history'].shape == (expected,300) and z['history'].dtype == np.float32
            assert np.isfinite(z['previous_action']).all() and np.isfinite(z['history']).all()
            context[key] = dict(rows=expected, prior_shape=list(z['previous_action'].shape), history_shape=list(z['history'].shape),
                prior_max_abs=float(np.abs(z['previous_action']).max()), history_max_abs=float(np.abs(z['history']).max()))
    manifest = read(PATHS['physical_manifest'])
    physical_context = {key: manifest['arrays'][key] for key in
        ('advanced_history','policy_actual_normalized_action','policy_applied_target','outgoing_raw_prior')}
    report = dict(saved_analysis_complete=True, input_sha256=request['input_sha256'], source_sha256=sha(__file__), request_sha256=sha(BASE/'request.json'),
        objectives=objective, per_cell_changes=per_cell, initial_gradient_evidence=read(PATHS['coefficient']),
        progress_columns=['nominal','full_state','physical','weighted_total','learning_rate','preclip_gradient_norm'],
        progress_windows=[dict(first_update=a+1,last_update=b,means=p[a:b].mean(0).tolist(),
            cell_means={k:v[a:b].mean(0).tolist() for k,v in curves.items()}) for a,b in windows],
        initial_optimizer_transient=dict(first_nominal=float(p[0,0]),second_nominal=float(p[1,0]),
            second_over_first=float(p[1,0]/p[0,0]),maximum_nominal_update=int(np.argmax(p[:,0])+1),
            maximum_preclip_gradient_norm=float(p[:,5].max()),gradient_clip_activated_rows=int(np.count_nonzero(p[:,5]>10))),
        late_response_window_fraction_change=float(p[9000:10000,1].mean()/p[8000:9000,1].mean()-1),
        response_comparison=baseline['comparison'],
        final_response_over_zero=baseline['results']['final_GPU32']['objective']/baseline['results']['zero_response']['objective'],
        final_response_odd_fraction=baseline['results']['final_GPU32']['odd']/baseline['results']['final_GPU32']['objective'],
        prior_hold_over_model=prior['projected_previous_action_balanced_MSE']/prior['final_model_balanced_MSE'],
        causal_context_availability=context, physical_context_manifest=physical_context,
        actual_failure=read(PATHS['outcome_report'])['failure'],
        model_calls=0, gradient_calls=0, optimizer_updates=0, native_steps=0,
        limitations=['Window averages use different fixed sampled pairs; flattening is not a convergence certificate.',
            'Only initial loss gradients were saved; no final or per-step conflict conclusion is available.',
            'Metadata availability does not replace a full context-index and physical prior audit.',
            'Poor prior-hold output does not measure information gained by conditioning a learned policy on prior/history.',
            'This analysis does not fit, select or qualify a new controller.'])
    for path,digest in request['input_sha256'].items():
        assert sha(path)==digest,path
    report['all_inputs_unchanged']=True
    write_new(BASE/'report.json',report)
    print(json.dumps(dict(report_sha256=sha(BASE/'report.json'),initial_spike_ratio=report['initial_optimizer_transient']['second_over_first'],
        per_cell_changes={k:{x:v[x] for x in ('improved','worsened')} for k,v in per_cell.items()},
        response_over_zero=report['final_response_over_zero'],prior_hold_over_model=report['prior_hold_over_model'],
        model_calls=0,native_steps=0)))

if __name__ == '__main__':
    main()
