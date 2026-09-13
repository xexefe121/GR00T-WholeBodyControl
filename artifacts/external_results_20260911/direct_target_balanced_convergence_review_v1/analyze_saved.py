"""Saved-only convergence diagnosis. Never imports Torch/ORT/MuJoCo."""
import hashlib
import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
NEW = OUT.parent
FIT = NEW / 'direct_target_causal_response_balanced_student_v2/fit'
INPUTS = {}
GROUPS = ['root_position', 'root_rotation', 'joint_position',
          'root_linear_velocity', 'root_angular_velocity', 'joint_velocity']
DATASETS = ['old', 'query1', 'query250', 'PICO', 'walk002']
PHASES = ['acquisition', 'middle', 'terminal']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def bind(path):
    path = Path(path).resolve()
    digest = sha(path)
    assert path.as_posix() not in INPUTS or INPUTS[path.as_posix()] == digest
    INPUTS[path.as_posix()] = digest
    return path


def read(path):
    return json.loads(bind(path).read_text(encoding='utf-8-sig'))


def array(name):
    path = bind(FIT / (name + '.npy'))
    assert sha(path) == manifest['files'][path.name]
    return np.load(path, allow_pickle=False, mmap_mode='r')


def write(name, value):
    with (OUT / name).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(value if isinstance(value, str) else json.dumps(value, indent=2, allow_nan=False) + '\n')


def comparison(before, after):
    return dict(initial=before, final=after, change_percent=100 * (after / before - 1))


report = read(FIT / 'report.json')
assert sha(FIT / 'report.json') == 'f918f1dfe675aae01e25307d302ce6c54f599fb7f0f5d1fa3fd943a179b5aded'
manifest = read(FIT / 'output_manifest.json')
request = read(FIT.parent / 'training_request.json')
owner = read(FIT.parent / 'owner_completion_verification.json')
assert owner['completion_passed'] and report['completed']
before, after, deployed = [report['metrics'][k] for k in ('initial_GPU32', 'final_GPU32', 'ORT64')]
objectives = {name: comparison(before[name], after[name]) for name in
              ('nominal_objective', 'full_state_objective', 'balanced_full_state_objective',
               'physical_objective', 'weighted_objective', 'balanced_weighted_objective')}
phase_results, cell_results, clipping = {}, {}, {}
for kind, loss in [('nominal', 'normalized_MSE'), ('full_state', 'response_MSE'), ('physical', 'response_MSE')]:
    initial_cells, final_cells = before[kind + '_cells'], after[kind + '_cells']
    assert len(initial_cells) == len(final_cells)
    cell_results[kind] = []
    for a, b in zip(initial_cells, final_cells):
        assert (a['dataset'], a['phase'], a.get('tangent_group')) == (b['dataset'], b['phase'], b.get('tangent_group'))
        row = dict(dataset=DATASETS[b['dataset']], phase=PHASES[b['phase']],
                   **comparison(a[loss], b[loss]),
                   preclip_RMSE_rad=comparison(a['preclip_RMSE_rad'], b['preclip_RMSE_rad']),
                   student_clipped_rows=[a['clipped_rows'], b['clipped_rows']],
                   student_clipped_components=[a['clipped_components'], b['clipped_components']])
        if kind == 'full_state':
            row.update(group=GROUPS[b['tangent_group']], zero_response_MSE=b['zero_response_MSE'],
                       response_over_zero=b[loss] / b['zero_response_MSE'])
        cell_results[kind].append(row)
    phase_results[kind] = {PHASES[i]: comparison(
        float(np.mean([a[loss] for a in initial_cells if a['phase'] == i])),
        float(np.mean([a[loss] for a in final_cells if a['phase'] == i]))) for i in range(3)}
    clipping[kind] = dict(initial_rows=sum(c['clipped_rows'] for c in initial_cells),
                         final_rows=sum(c['clipped_rows'] for c in final_cells),
                         initial_components=sum(c['clipped_components'] for c in initial_cells),
                         final_components=sum(c['clipped_components'] for c in final_cells),
                         corpus_rows={'nominal': 9904, 'full_state': 354612, 'physical': 3054}[kind])
groups = []
for index, (a, b) in enumerate(zip(before['full_state_group_comparison'], after['full_state_group_comparison'])):
    groups.append(dict(group=GROUPS[index], **comparison(a['response_MSE'], b['response_MSE']),
                       zero_response_MSE=b['zero_response_MSE'], response_over_zero=b['response_MSE'] / b['zero_response_MSE']))
progress = array('training_progress')
original = array('original_objectives')
assert progress.shape == (3000, 6) and original.shape == (3000, 2)
windows = []
for low, high in [(0, 250), (250, 500), (500, 1000), (1000, 1500),
                  (1500, 2000), (2000, 2500), (2500, 2750), (2750, 3000)]:
    row = dict(update_start=low + 1, update_end=high,
               **dict(zip(report['training_loss_columns'], progress[low:high].mean(axis=0).tolist())),
               original_full_state=float(original[low:high, 0].mean()),
               original_total=float(original[low:high, 1].mean()),
               maximum_gradient_norm=float(progress[low:high, 5].max()))
    windows.append(row)
ledger_cells = {}
for kind in ('nominal', 'full_state', 'physical', 'balanced_full_state'):
    value = array(kind + '_cell_losses')
    ledger_cells[kind] = dict(first250_mean=value[:250].mean(axis=0).tolist(),
                              last250_mean=value[-250:].mean(axis=0).tolist(),
                              first=value[0].tolist(), last=value[-1].tolist())
with np.load(bind(FIT / 'shared/normalization.npz'), allow_pickle=False) as z:
    default, span, limits = z['default_q'].copy(), z['joint_span'].astype(np.float64), z['joint_limits'].copy()
with np.load(bind(request['paths']['centers']), allow_pickle=False) as z:
    matches = np.flatnonzero((z['dataset'] == 2) & (z['control'] == 250))
    assert matches.tolist() == [2038]
    target = z['expert_target'][2038].copy()
contract = read(request['paths']['contract'])
entry = dict(center_row=2038, dataset='query250', control=250, source_frame=261,
             teacher_target=target.tolist(), source='saved nominal row; no actual71000 rollout or new batch1 call')
for backend in ('initial_GPU32', 'final_GPU32', 'ORT64'):
    head = array(backend + '_nominal')[2038].astype(np.float64)
    raw = default + span * head
    applied = np.clip(raw, limits[:, 0], limits[:, 1])
    error = raw - target
    entry[backend] = dict(preclip_RMSE_rad=float(np.sqrt(np.mean(error ** 2))),
                          applied_RMSE_rad=float(np.sqrt(np.mean((applied - target) ** 2))),
                          maximum_error_rad=float(np.max(np.abs(error))),
                          worst_joint=contract['joint_names'][int(np.argmax(np.abs(error)))],
                          clipped_components=int(np.count_nonzero(raw != applied)),
                          raw_target=raw.tolist(), applied_target=applied.tolist(),
                          per_joint_error_rad=error.tolist())
prior_trace = bind(NEW / 'direct_target_causal_context_evaluation_v2/nominal/trace.npz')
assert sha(prior_trace) == '589cbab8051ab9c1257049d62098248e7866b10e6c0eb085e73cbaf32d79082d'
with np.load(prior_trace, allow_pickle=False) as z:
    inference_ms = z['inference_ms'][250:303].copy()
assert inference_ms.shape == (53,)
timing = dict(source='prior causal68000 canonical saved53 learned inference_ms values',
              count=53, minimum_ms=float(inference_ms.min()), median_ms=float(np.median(inference_ms)),
              p95_ms=float(np.quantile(inference_ms, .95)), maximum_ms=float(inference_ms.max()),
              current_dense_MACs=1323*256+256*256+256*23,
              proposed_width512_dense_MACs=1323*512+512*512+512*23,
              not_full_controller_or_hard_deadline_qualification=True)
result = dict(
    saved_diagnosis_completed=True, ordinary_step=71000, comparison_backend='GPU32 initial versus final',
    deployed_backend_separately_reported='ORT64', objectives=objectives,
    balanced_response_over_zero=after['balanced_full_state_objective']/after['balanced_full_state_zero_response_MSE'],
    original_response_over_zero=after['full_state_objective']/after['full_state_zero_response_MSE'],
    balanced_odd_error_fraction=after['balanced_full_state_odd_MSE']/after['balanced_full_state_objective'],
    original_odd_error_fraction=after['full_state_odd_MSE']/after['full_state_objective'],
    phases=phase_results, cells=cell_results, groups=groups, student_clipping=clipping,
    teacher_flag_diagnostics=after['full_state_flag_diagnostics'],
    training_windows=windows, training_cell_windows=ledger_cells,
    gradient=dict(maximum=float(progress[:,5].max()), maximum_update=int(progress[:,5].argmax()+1),
                  median=float(np.median(progress[:,5])), clipped_update_count=int(np.count_nonzero(progress[:,5]>10))),
    transient=dict(nominal_peak_update=int(progress[:,0].argmax()+1),
                   nominal_peak_MSE=float(progress[:,0].max()),
                   nominal_peak_over_initial=float(progress[:,0].max()/progress[0,0])),
    entry=entry, timing=timing,
    limitations=[
        'These are saved supervised-training rows, not independent generalization data or a new rollout.',
        'Full-state training windows use different fixed Monte Carlo batches; their fluctuations are not fixed-corpus checkpoint comparisons.',
        'One scalar total gradient norm cannot identify individual loss conflict, activation saturation, rank, or capacity.',
        'Odd error measures directional target-change mismatch; it does not prove the student response is simply too small or has the wrong sign.',
        'Clipping counts refer to raw proposals beyond native target bounds; they are not native physics limit failures.',
        'Teacher maps are committed local feedback maps; these data do not establish replanned actual-state expert optimality.',
        'The71000 live canonical has not been executed by this diagnostic.'],
    input_sha256=INPUTS, source_sha256=sha(__file__),
    model_calls=0, gradient_calls=0, optimizer_updates=0, native_steps=0)
for path, digest in INPUTS.items():
    assert sha(path) == digest, path
write('report.json', result)
write('evidence_summary.md', f"""Saved71000 fit improves overall errors, but leaves a large directional-control mismatch. This diagnosis used saved ledgers and predictions only; no new model or simulation calls.

InitialGPU32→finalGPU32 nominal loss improves3.52%, original full-state response2.34%, balanced response8.95%, physical response4.64%. All15 nominal and9 physical cells improve;49/54 response cells improve. Balanced response remains{result['balanced_response_over_zero']:.6f}× the zero-response baseline; original response remains{result['original_response_over_zero']:.6f}×. Directional odd error is{100*result['balanced_odd_error_fraction']:.2f}% of balanced error. A zero-response baseline is a diagnostic, not a proposed stabilizing controller.

| Tangent group | Response change | Final error / zero response |
|---|---:|---:|
""" + ''.join(f"| {g['group']} | {g['change_percent']:+.2f}% | {g['response_over_zero']:.3f}× |\n" for g in groups) + f"""
Acquisition nominal loss improves8.38%; middle phase only0.97%; terminal5.93%. PICO middle remains the largest nominal cell, MSE0.00140924. These equal-cell averages do not show a missing phase in the sampler. Widening phase weights alone would trade existing cells against each other without directly repairing local response representation.

The exact query250/control250 saved nominal entry gets worse:0.0473776621→0.0479952678rad RMSE on GPU32. FinalORT64 is0.0479951952rad; no target components clip there. Left hip pitch remains the largest error,0.153246rad. Query250 first24 acquisition rows improve0.055837→0.053535rad, so even this small regional average hides the worsened first action. This is nominal evidence, not a prediction of the next actual failure time.

Native-target clipping decreases modestly: nominal5011→4946/9904 rows; full-state176448→173615/354612; physical1522→1500/3054. Teacher full-state native-clipped flags cover19298 rows; feedback-clipped flags64677. Student and teacher mask semantics are retained separately. Saturation remains widespread, but the first-entry error exists before saturation.

Warm LR restart produces a nominal-loss peak at update5,6.74× the initial batch loss, before recovery. Total gradient norm peaks0.555806, below clip10; clipping never limits optimization. After update250, norms remain around0.0012–0.0014. Late nominal and physical losses still decline slowly as LR approaches1e-6. Changing sampled response batches prevents reading the last two response-window means as a fixed-corpus plateau. There is no evidence here for raising the clipping ceiling or using scalar loss magnitude to select another coefficient.

The saved evidence establishes supervised underfit of directional responses. It cannot distinguish insufficient capacity from objective interference or insufficient optimization. Correct causal-history alignment and the prior matched context improvement weaken a pure bookkeeping explanation. Wider response coverage alone also has not solved the problem. Actual off-trajectory states may exceed these local single-axis neighborhoods; no capacity change certifies stability there.

If the71000 canonical fails, prepare one function-preserving width512 capacity experiment, keeping causal1323 inputs, all qualified labels, normalization, balanced coefficients and strict acceptance. Preserve old256 neuron blocks and optimizer moments; add seeded nonsymmetric incoming units with zero outgoing paths. This gives a concrete representational change while retaining the old controller at initialization. It is an engineering trial, not proof that capacity was the unique cause. Do not simultaneously reweight phases, retune derivatives, change PD gains, or select an intermediate rollout winner.

Use a meaningful fixed optimization proposal rather than another tiny low-rate continuation:10000 updates, retain warm old moments,250-step LR ramp1e-6→1e-5 followed by9750-step cosine1e-5→1e-6. The ramp responds to the observed warm-restart transient; fixed endpoint and independent saved metrics still govern evaluation. These optimizer details are a proposal requiring separate source/request review, not a selected fit. Synthetic expansion tests must prove original functions, all six optimizer states and the new gradient path before any actual checkpoint use.

Current1323→256→256→23 uses410112 dense MACs; width512 uses951296,2.32×. Prior68000's53 learned inference timings were median{timing['median_ms']:.3f}ms,p95{timing['p95_ms']:.3f}ms,max{timing['maximum_ms']:.3f}ms. Those are inference measurements, not full controller deadlines, and linear extrapolation does not qualify the wider model. Require measured WSLbatch1 feature construction plus head+target processing below20ms before deployment, unchanged FP64/publicFP32 1e-5rad export checks, then original1569+conditional250 native acceptance. No hardware or connected clock qualification follows from this diagnosis.

Exact per-cell values, clipping counts, window means, entry vectors, input hashes and limitations: [report.json](report.json).
""")
write('completion.json', dict(completed=True, report_sha256=sha(OUT/'report.json'),
                              summary_sha256=sha(OUT/'evidence_summary.md'),
                              source_sha256=sha(__file__), input_count=len(INPUTS),
                              model_calls=0, gradient_calls=0, optimizer_updates=0, native_steps=0))
print(json.dumps(dict(report_sha256=sha(OUT/'report.json'), summary_sha256=sha(OUT/'evidence_summary.md'))))
