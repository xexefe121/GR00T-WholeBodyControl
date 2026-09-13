"""Saved causal-input diagnostic only. Does not substitute a controller."""
import json
from pathlib import Path
import numpy as np
from analyze_saved import BASE, NEW, FIT, PINS, sha, bind, read

def run():
    root = FIT.parent
    request = read(root / 'training_request.json', '0c3bbc4e8567f6dfc09dfe7d9cdfcdd8af5a56178ea112af3a647d84c48d0dea')
    frozen = read(root / 'training_frozen_inputs.json', 'bbb67b48187dcead0dc0886496573447ffdd58582eaeb63f41a7fa947e922f18')
    known = {Path(p).resolve(): d for p, d in frozen['input_sha256'].items()}
    paths = request['paths']
    contract_path = Path(paths['contract']).resolve()
    contract = read(contract_path, known[contract_path])
    default = np.asarray(contract['default_q'], np.float64)
    limits = np.asarray(contract['joint_limits'], np.float64)
    span = np.diff(limits, axis=1).ravel().astype(np.float32).astype(np.float64)
    prior_targets, teachers = [], []
    for key in ('centers', 'pico', 'walk002'):
        path = Path(paths[key]).resolve()
        with np.load(bind(path, known[path]), allow_pickle=False) as data:
            prior = data['previous_action']
            previous = np.clip(default + prior * .25 * np.asarray(contract['training_effort']) / np.asarray(contract['kp']), limits[:, 0], limits[:, 1])
            expected = (previous - default).astype(np.float32)
            assert data['features'][:, 52:75].tobytes() == expected.tobytes()
            prior_targets.append(previous)
            teachers.append(data['expert_target'].copy())
    prior_targets = np.concatenate(prior_targets)
    teacher = np.concatenate(teachers)
    assert prior_targets.shape == teacher.shape == (9904, 23)
    baseline = ((prior_targets - default) / span).astype(np.float32)
    label = ((teacher - default) / span).astype(np.float32)
    error = np.square(baseline - label).astype(np.float64)
    report = read(FIT / 'report.json', '7426d281be79c3610ef02ae8889c3ef66264da15a9667551d6bca4166a4beaea')
    counts = ((100, 819, 100), (100, 819, 100), (100, 819, 100), (100, 5780, 100), (100, 667, 100))
    cursor = 0
    cells = []
    for dataset, phases in enumerate(counts):
        for phase, n in enumerate(phases):
            index = dataset * 3 + phase
            rows = slice(cursor, cursor + n)
            value = float(error[rows].mean())
            trained = report['metrics']['final_GPU32']['nominal_cells'][index]
            cells.append(dict(dataset=dataset, phase=phase, rows=n, previous_action_projected_normalized_MSE=value,
                previous_action_projected_RMSE_rad=float(np.sqrt(np.square(prior_targets[rows] - teacher[rows]).mean())),
                final_model_normalized_MSE=trained['normalized_MSE'], final_model_RMSE_rad=trained['preclip_RMSE_rad']))
            cursor += n
    assert cursor == 9904
    for path, digest in PINS.items():
        assert sha(path) == digest
    result = dict(saved_analysis_complete=True, cells=cells, input_sha256=PINS, source_sha256=sha(__file__),
        projected_previous_action_balanced_MSE=float(np.mean([c['previous_action_projected_normalized_MSE'] for c in cells])),
        final_model_balanced_MSE=report['metrics']['final_GPU32']['nominal_objective'],
        existing_feature_columns_exact=True, model_calls=0, native_steps=0, optimizer_updates=0,
        limitations=['This uses the recorded expert previous raw action projected through existing native target reconstruction.',
            'Current direct model omits these existing causal feature columns52:75; predictive utility does not prove causation.',
            'Holding a previous target is not qualified as a tracking controller and provides no perturbation recovery claim.',
            'Autoregressive input or output changes would require new training and independent closed-loop evaluation.'])
    with (BASE / 'previous_target_report.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(baseline=result['projected_previous_action_balanced_MSE'], final=result['final_model_balanced_MSE'],
        query250=cells[6:9], report_sha256=sha(BASE / 'previous_target_report.json'))))

if __name__ == '__main__':
    run()
