"""Compare immutable 71000/81000 predictions and ledgers; zero task execution."""
import hashlib
import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
ROOTS = {71000: BASE / 'direct_target_causal_response_balanced_student_v2',
         81000: BASE / 'direct_target_causal_width512_student_v1'}
EXPECTED = {71000: 'f918f1dfe675aae01e25307d302ce6c54f599fb7f0f5d1fa3fd943a179b5aded',
            81000: 'b5979023fe9bceb14e0d54cf04bf727fbc413d5a1443b39018e149443b418de6'}
GROUPS = ['root_position', 'root_rotation', 'joint_position', 'root_linear_velocity',
          'root_angular_velocity', 'joint_velocity']
PHASES = ['acquisition', 'middle', 'terminal']
DATASETS = ['old', 'query1', 'query250', 'PICO', 'walk002']
INPUTS = {}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def bind(path, expected=None):
    path = Path(path).resolve()
    digest = sha(path)
    if expected is not None:
        assert digest == expected, path
    assert str(path) not in INPUTS or INPUTS[str(path)] == digest
    INPUTS[str(path)] = digest
    return path


def read(path, expected=None):
    return json.loads(bind(path, expected).read_text(encoding='utf-8-sig'))


def write(name, value):
    with (OUT / name).open('x', encoding='utf-8', newline='\n') as f:
        f.write(value if isinstance(value, str) else json.dumps(value, indent=2, allow_nan=False) + '\n')


def comp(old, new):
    return dict(before=float(old), after=float(new), change_percent=float(100 * (new / old - 1)))


reports, manifests, requests, owners = {}, {}, {}, {}
for step, root in ROOTS.items():
    reports[step] = read(root / 'fit/report.json', EXPECTED[step])
    manifests[step] = read(root / 'fit/output_manifest.json')
    requests[step] = read(root / 'training_request.json')
    owners[step] = read(root / 'owner_completion_verification.json')
    assert reports[step]['completed'] and owners[step]['completion_passed']


def saved(step, name):
    path = ROOTS[step] / 'fit' / name
    assert name in manifests[step]['files'], name
    return bind(path, manifests[step]['files'][name])


def array(step, name):
    return np.load(saved(step, name + '.npy'), mmap_mode='r', allow_pickle=False)


audit = read(BASE / 'direct_target_width512_fit_independent_v1/results_v1/report.json',
             '3656f44031d9362fdbabeab69d2423266839b88662cbe46b79953a977dc56cd7')
assert audit['passed'] and audit['export_qualified']
read(BASE / 'direct_target_width512_fit_independent_v1/owner_completion.json',
     '24b9ba22d572874a5e4bd4c97a282e9444126231807ff70d526312303eb12122')
coefficient_subject = requests[81000]['subjects']['coefficient_source']
read(coefficient_subject['path'], coefficient_subject['sha256'])
COEF = 1.8188207859141674
OBJECTIVES = ['nominal_objective', 'full_state_objective', 'balanced_full_state_objective',
              'physical_objective', 'weighted_objective', 'balanced_weighted_objective']
comparisons = {}
for backend in ['final_GPU32', 'ORT64']:
    metrics = {}
    for step in ROOTS:
        metrics[step] = json.loads(saved(step, backend + '_metrics.json').read_text())
        assert metrics[step] == reports[step]['metrics'][backend]
    old, new = metrics[71000], metrics[81000]
    objectives = {key: comp(old[key], new[key]) for key in OBJECTIVES}
    cells, phases, clipping = {}, {}, {}
    for kind, loss in [('nominal', 'normalized_MSE'), ('full_state', 'response_MSE'),
                       ('physical', 'response_MSE'), ('balanced_full_state', 'weighted_response_MSE')]:
        prior, final = old[kind + '_cells'], new[kind + '_cells']
        cells[kind] = []
        for a, b in zip(prior, final):
            assert (a['dataset'], a['phase'], a.get('tangent_group')) == (b['dataset'], b['phase'], b.get('tangent_group'))
            row = dict(dataset=DATASETS[b['dataset']], phase=PHASES[b['phase']], **comp(a[loss], b[loss]))
            if 'tangent_group' in b:
                row['group'] = GROUPS[b['tangent_group']]
                zero_key = 'weighted_zero_response_MSE' if kind == 'balanced_full_state' else 'zero_response_MSE'
                row['zero_response_MSE'] = b[zero_key]
                row['response_over_zero'] = b[loss] / b[zero_key]
            if 'preclip_RMSE_rad' in b:
                row['preclip_RMSE_rad'] = comp(a['preclip_RMSE_rad'], b['preclip_RMSE_rad'])
                row['applied_RMSE_rad'] = comp(a['applied_RMSE_rad'], b['applied_RMSE_rad'])
                row['clipped_rows'] = [a['clipped_rows'], b['clipped_rows']]
            cells[kind].append(row)
        phases[kind] = {PHASES[i]: comp(np.mean([c[loss] for c in prior if c['phase'] == i]),
                                             np.mean([c[loss] for c in final if c['phase'] == i])) for i in range(3)}
        if kind != 'balanced_full_state':
            clipping[kind] = dict(before_rows=sum(c['clipped_rows'] for c in prior),
                                  after_rows=sum(c['clipped_rows'] for c in final),
                                  before_components=sum(c['clipped_components'] for c in prior),
                                  after_components=sum(c['clipped_components'] for c in final),
                                  corpus_rows={'nominal': 9904, 'full_state': 354612, 'physical': 3054}[kind])
        assert np.isclose(np.mean([c[loss] for c in final]), new[kind + '_objective'], rtol=1e-12, atol=1e-16)
    assert np.isclose(new['nominal_objective'] + COEF * new['balanced_full_state_objective'] +
                      new['physical_objective'], new['balanced_weighted_objective'], rtol=1e-14)
    groups = []
    for i, (a, b) in enumerate(zip(old['full_state_group_comparison'], new['full_state_group_comparison'])):
        groups.append(dict(group=GROUPS[i], **comp(a['response_MSE'], b['response_MSE']),
                           zero_response_MSE=b['zero_response_MSE'],
                           before_over_zero=a['response_MSE']/a['zero_response_MSE'],
                           after_over_zero=b['response_MSE']/b['zero_response_MSE'],
                           weighted_response_MSE=new['balanced_full_state_group_comparison'][i]['weighted_response_MSE']))
    comparisons[backend] = dict(objectives=objectives, groups=groups, phases=phases, cells=cells,
        clipping=clipping, improved_cells={k: sum(c['after'] < c['before'] for c in v) for k, v in cells.items()},
        response_over_zero=comp(old['full_state_objective']/old['full_state_zero_response_MSE'],
                                new['full_state_objective']/new['full_state_zero_response_MSE']),
        balanced_response_over_zero=comp(old['balanced_full_state_objective']/old['balanced_full_state_zero_response_MSE'],
                                         new['balanced_full_state_objective']/new['balanced_full_state_zero_response_MSE']),
        odd_error_fraction=new['full_state_odd_MSE']/new['full_state_objective'],
        balanced_odd_error_fraction=new['balanced_full_state_odd_MSE']/new['balanced_full_state_objective'],
        original_odd_error=comp(old['full_state_odd_MSE'], new['full_state_odd_MSE']),
        original_even_error=comp(old['full_state_even_MSE'], new['full_state_even_MSE']),
        balanced_odd_error=comp(old['balanced_full_state_odd_MSE'], new['balanced_full_state_odd_MSE']),
        balanced_even_error=comp(old['balanced_full_state_even_MSE'], new['balanced_full_state_even_MSE']),
        teacher_flags=new['full_state_flag_diagnostics'])

# Exactly the runtime cast contract: stored span f32 promoted to f64, default f64,
# each saved public f32 head promoted before multiplication; then native f64 clamp.
norms = []
for step in ROOTS:
    with np.load(bind(ROOTS[step]/'fit/shared/normalization.npz'), allow_pickle=False) as z:
        norms.append({key: z[key].copy() for key in z.files})
assert norms[0].keys() == norms[1].keys()
for key in norms[0]:
    assert norms[0][key].dtype == norms[1][key].dtype and norms[0][key].tobytes() == norms[1][key].tobytes(), key
default = norms[1]['default_q']; span = norms[1]['joint_span'].astype(np.float64); limits = norms[1]['joint_limits']
with np.load(bind(requests[81000]['paths']['centers']), allow_pickle=False) as z:
    assert np.flatnonzero((z['dataset'] == 2) & (z['control'] == 250)).tolist() == [2038]
    targets = z['expert_target'][2038:2138].copy()
    assert z['control'][2038:2138].tolist() == list(range(250, 350))
contract = read(requests[81000]['paths']['contract'])
entry = {}
for backend in ['final_GPU32', 'ORT64']:
    entry[backend] = {}
    for step in ROOTS:
        head = array(step, backend + '_nominal')[2038:2138]
        assert head.shape == (100, 23) and head.dtype == np.float32
        raw = default + span * head.astype(np.float64)
        applied = np.clip(raw, limits[:, 0], limits[:, 1])
        value = {}
        for count in [1, 24, 100]:
            error = raw[:count] - targets[:count]
            value[str(count)] = dict(preclip_RMSE_rad=float(np.sqrt(np.mean(error**2))),
                                     applied_RMSE_rad=float(np.sqrt(np.mean((applied[:count]-targets[:count])**2))),
                                     maximum_absolute_error_rad=float(np.abs(error).max()),
                                     clipped_rows=int(np.any(raw[:count] != applied[:count], axis=1).sum()),
                                     clipped_components=int((raw[:count] != applied[:count]).sum()),
                                     per_joint_RMSE_rad=np.sqrt(np.mean(error**2, axis=0)).tolist())
        value['first_action_error_rad'] = (raw[0]-targets[0]).tolist()
        value['first_action_target'] = raw[0].tolist()
        value['worst_first_joint'] = contract['joint_names'][int(np.argmax(np.abs(raw[0]-targets[0])))]
        cells = reports[step]['metrics'][backend]['nominal_cells']
        cell = next(c for c in cells if c['dataset'] == 2 and c['phase'] == 0)
        assert np.isclose(value['24']['preclip_RMSE_rad'], cell['first24']['preclip_RMSE_rad'], rtol=1e-12, atol=1e-15)
        assert np.isclose(value['100']['preclip_RMSE_rad'], cell['preclip_RMSE_rad'], rtol=1e-12, atol=1e-15)
        entry[backend][str(step)] = value
    entry[backend]['comparisons'] = {str(n): comp(entry[backend]['71000'][str(n)]['preclip_RMSE_rad'],
                                               entry[backend]['81000'][str(n)]['preclip_RMSE_rad']) for n in [1,24,100]}

ledger = array(81000, 'training_progress')
original = array(81000, 'original_objectives')
assert ledger.shape == (10000, 6) and original.shape == (10000, 2)
assert np.isfinite(ledger).all() and np.isfinite(original).all()
windows = []
for low, high in [(0,50),(50,250),(250,500),(500,1000),(1000,2000),(2000,4000),
                   (4000,6000),(6000,8000),(8000,9000),(9000,9750),(9750,10000)]:
    windows.append(dict(update_first=low+1, update_last=high,
        means=dict(zip(reports[81000]['training_loss_columns'], ledger[low:high].mean(axis=0).tolist())),
        original_full_state=float(original[low:high,0].mean()), original_total=float(original[low:high,1].mean()),
        max_grad_norm=float(ledger[low:high,5].max()), max_nominal=float(ledger[low:high,0].max())))
training_cells = {}
for kind in ['nominal', 'full_state', 'physical', 'balanced_full_state']:
    values = array(81000, kind + '_cell_losses')
    assert values.shape == (10000, {'nominal':15, 'full_state':54, 'physical':9, 'balanced_full_state':54}[kind])
    training_cells[kind] = {name: values[a:b].mean(axis=0).tolist() for name,a,b in
                           [('first250',0,250),('middle1000',4500,5500),('last1000',9000,10000),('last250',9750,10000)]}
gradient = dict(maximum=float(ledger[:,5].max()), maximum_update=int(ledger[:,5].argmax()+1),
                median=float(np.median(ledger[:,5])), p95=float(np.quantile(ledger[:,5],.95)),
                clipped_updates=int(np.count_nonzero(ledger[:,5] > 10)))
transient = dict(first_nominal=float(ledger[0,0]), peak_nominal=float(ledger[:,0].max()),
                 peak_update=int(ledger[:,0].argmax()+1), peak_over_first=float(ledger[:,0].max()/ledger[0,0]),
                 rates_at_selected_updates={str(i+1): float(ledger[i,4]) for i in [0,249,250,9999]})
limitations = [
    'Saved supervised training corpora only; no new model, gradient, optimizer, native step or clock.',
    'Capacity, 10000 extra updates, LR ramp and warm-state expansion were not separated by a matched width256 continuation.',
    'Whole-corpus final endpoints are comparable; changing sampled full-state batches prevent reading ledger windows as fixed-corpus checkpoint curves.',
    'Odd/even residuals alone do not identify predicted response gain, sign, loss-gradient conflict or a unique failure cause.',
    'Zero response is a sensitivity diagnostic, not a proposed stable controller or proof that an aggregate below it is sufficient.',
    'Raw target clipping is distinct from a native speed/range failure; teacher feedback and native target clipping remain separate.',
    'Single-axis local fixed-map labels do not prove correctness on coupled off-trajectory states or replanned expert optimality.',
    'No actual81000 canonical outcome used. Width512 latency, full segment and end hold require their own actual qualification.']
for path, digest in INPUTS.items():
    assert sha(path) == digest, path
result = dict(passed=True, saved_diagnosis_completed=True, before_step=71000, after_step=81000,
    comparison_backends_separate=['final_GPU32','ORT64'], comparisons=comparisons,
    query250=dict(center_row=2038, controls=[250,349], teacher_first_target=targets[0].tolist(), backends=entry),
    training_windows=windows, training_cell_windows=training_cells, gradient=gradient, transient=transient,
    independent_fit_audit_passed=True, independent_fit_audit_checks=int(audit['checks']),
    limitations=limitations, input_sha256=INPUTS, source_sha256=sha(__file__),
    model_calls=0, ORT_calls=0, gradient_calls=0, optimizer_updates=0, native_steps=0)
write('report.json', result)
c = comparisons['ORT64']; o = c['objectives']; e = entry['ORT64']; new_entry = e['81000']
lines = [
    'Width512 trial substantially improves saved fit; response mismatch remains. This compares71000 and81000 ordinary endpoints using the same ORT64 backend. GPU32 comparisons are retained separately. No new model or simulation calls.\n',
    f"Nominal MSE {o['nominal_objective']['before']:.10g} → {o['nominal_objective']['after']:.10g} ({o['nominal_objective']['change_percent']:+.2f}%); physical response {o['physical_objective']['change_percent']:+.2f}%; original full-state response {o['full_state_objective']['change_percent']:+.2f}%; balanced response {o['balanced_full_state_objective']['change_percent']:+.2f}%; selected total {o['balanced_weighted_objective']['change_percent']:+.2f}%. Improved cells: nominal {c['improved_cells']['nominal']}/15, physical {c['improved_cells']['physical']}/9, response {c['improved_cells']['full_state']}/54.\n",
    f"Original response/zero falls {c['response_over_zero']['before']:.6f}× → {c['response_over_zero']['after']:.6f}×. Balanced response/zero falls {c['balanced_response_over_zero']['before']:.6f}× → {c['balanced_response_over_zero']['after']:.6f}× and remains worse than zero. Balanced odd error contributes {100*c['balanced_odd_error_fraction']:.2f}%; this establishes directional mismatch, not its sign or cause.\n",
    '| Tangent group | Error change | Final error / zero |\n|---|---:|---:|',
    *[f"| {g['group']} | {g['change_percent']:+.2f}% | {g['after_over_zero']:.4f}× |" for g in c['groups']],
    f"\nExact query250/control250 saved entry improves {e['comparisons']['1']['before']:.10f} → {e['comparisons']['1']['after']:.10f} rad RMSE ({e['comparisons']['1']['change_percent']:+.2f}%). No first-action clipping: {new_entry['1']['clipped_components']} components; largest error {new_entry['1']['maximum_absolute_error_rad']:.6f} rad at {new_entry['worst_first_joint']}. First24 acquisition rows improve {e['comparisons']['24']['before']:.8f} → {e['comparisons']['24']['after']:.8f} rad. Whole100-row acquisition improves {e['comparisons']['100']['change_percent']:+.2f}%. This uses saved teacher-state inputs; actual departed states remain untested here.\n",
    'Raw native-target-clipped rows: ' + '; '.join(f"{k} {v['before_rows']} → {v['after_rows']} / {v['corpus_rows']}" for k,v in c['clipping'].items()) + '. These counts do not measure actual physics violations.\n',
    f"Nominal training loss peaks at update {transient['peak_update']}, {transient['peak_over_first']:.3f}× first-batch loss. Total preclip gradient norm maximum {gradient['maximum']:.6g}, median {gradient['median']:.6g}; {gradient['clipped_updates']} updates exceed clip10. Ramp and later window values are retained in report.json. Full-state sampled-window fluctuations do not establish a fixed-corpus plateau.\n",
    'The engineering change improves representational fit while retaining the old function at initialization. It does not uniquely prove a capacity bottleneck: width,10000 more updates, ramped LR and new moments all changed together. Weakest residual sensitivity remains joint velocity, followed by root/joint position. No evidence here supports raising gradient clipping, replacing strict physics gates, or declaring a stable full-body controller.\n',
    'No further fit selected. Use the already selected canonical result to decide next work. If it fails, distinguish first-action nominal error from early feedback mismatch using actual saved departures and all58 tangent groups. A concrete follow-up should target whichever residual that trace supports; these supervised improvements alone cannot choose more width, changed loss weights or broader coupled-state labels. Require measured feature+inference+target latency below20ms and original1569+conditional250 acceptance separately.\n',
    'Exact objectives, all cells/phases/groups, query250 errors, clipping and training windows: [report.json](report.json).']
write('diagnosis.md', '\n'.join(lines) + '\n')
write('completion.json', dict(passed=True, report_sha256=sha(OUT/'report.json'),
    diagnosis_sha256=sha(OUT/'diagnosis.md'), source_sha256=sha(__file__), input_count=len(INPUTS),
    model_calls=0, ORT_calls=0, gradient_calls=0, optimizer_updates=0, native_steps=0))
print(json.dumps(dict(report_sha256=sha(OUT/'report.json'), diagnosis_sha256=sha(OUT/'diagnosis.md'),
                     ORT64_objectives=o, query250=e['comparisons'], gradient=gradient, transient=transient)))
