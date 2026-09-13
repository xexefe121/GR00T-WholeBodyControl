"""Pure saved-array validation for the selected two-clip collection."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from gear_sonic.utils.g1_true23_bfm_seed_observations import BFMHistory, state_and_terms

KEYS = ('actions', 'base_ang_vel', 'dof_pos', 'dof_vel', 'projected_gravity')
SHAPES = dict(features=(1069,), residual_rad=(23,), base_target=(23,), expert_target=(23,),
              base_action=(23,), previous_action=(23,), history=(300,), state=(52,),
              source_frame=(), control=(), phase=())
INTEGER = {'source_frame', 'control', 'phase'}
FLOAT32 = {'features', 'base_action', 'previous_action', 'history', 'state'}


def local(path):
    text = str(path).replace('\\', '/')
    if sys.platform != 'win32' and len(text) > 2 and text[1] == ':':
        text = '/mnt/' + text[0].lower() + text[2:]
    return Path(text)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def archive(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def atomic_write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    write(temporary, value)
    temporary.replace(path)


def exact(actual, expected, label):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
        raise ValueError('Byte mismatch: ' + label)


def action_from_target(target, contract):
    c = {key: np.asarray(contract[key], dtype=np.float64)
         for key in ('default_q', 'kp', 'training_effort')}
    return ((target - c['default_q']) * c['kp'] / (.25 * c['training_effort'])).astype(np.float32)


def phase_code(control, phase):
    for index, name in enumerate(('acquisition_ramp', 'source_motion', 'return_ramp')):
        if phase[name]['control_start'] <= control < phase[name]['control_stop']:
            return index
    raise ValueError('Unselected moving control: ' + str(control))


def moving_phases(timeline, clip):
    expected = dict(pico=(6530, 6230, 5980, 5780), walk002=(1417, 1117, 867, 667))
    if clip not in expected:
        raise ValueError('Unselected clip: ' + str(clip))
    total, stop, count, source_count = expected[clip]
    phases = {p['name']: p for p in timeline['phases']}
    assert timeline['total_requested_controls'] == total
    assert phases['initial_standing']['control_start'] == 0
    assert phases['initial_standing']['control_stop'] == 250
    assert phases['acquisition_ramp']['control_start'] == 250
    assert phases['acquisition_ramp']['control_stop'] == 350
    assert phases['source_motion']['control_start'] == 350
    assert phases['source_motion']['control_stop'] == stop - 100
    assert phases['source_motion']['requested_controls'] == source_count
    assert phases['return_ramp']['control_start'] == stop - 100
    assert phases['return_ramp']['control_stop'] == stop
    assert phases['returned_standing']['control_start'] == stop
    assert phases['returned_standing']['control_stop'] == total - 50
    assert phases['standing_proof_margin']['control_start'] == total - 50
    assert phases['standing_proof_margin']['control_stop'] == total
    assert stop - 250 == count
    for name in ('acquisition_ramp', 'source_motion', 'return_ramp'):
        assert phases[name]['requested_controls'] == phases[name]['control_stop'] - phases[name]['control_start']
    return phases, total, stop


def measured_rows(trace, contract, history_key, previous_key, end_control_inclusive):
    """Yield actual precontrol observations; never infer or step a simulator."""
    history = BFMHistory()
    previous = np.zeros(23, np.float32)
    default = np.asarray(contract['default_q'], np.float64)
    for control in range(end_control_inclusive + 1):
        q, dq = trace['qpos'][control], trace['qvel'][control]
        exact(previous, trace[previous_key][control], 'previous at ' + str(control))
        state, terms = state_and_terms(q[7:], dq[6:], q[3:7], dq[3:6], previous, default)
        named = {key: history.data[key].copy() for key in KEYS}
        flat = history.before_update(terms)
        exact(flat, trace[history_key][control], 'history at ' + str(control))
        if 'state' in trace:
            exact(state, trace['state'][control], 'state at ' + str(control))
        yield dict(control=control, qpos=q, qvel=dq, previous_action=previous.copy(),
                   state=state, history=flat, named_history=named)
        if control < end_control_inclusive:
            previous = action_from_target(trace['target'][control], contract)


def check_frozen(manifest_path):
    manifest = read(manifest_path)
    assert manifest['root_selected_extraction'] is True
    assert manifest['fitting_authorized'] is False
    assert manifest['collector_physics_authorized'] is False
    assert manifest['selected_rows'] == 6847
    assert [c['clip'] for c in manifest['cases']] == ['pico', 'walk002']
    assert manifest['baseline_goal'] == 'original_h8_position1_yaw2'
    assert manifest['feature_count'] == 1069
    for name, digest in manifest['input_hashes'].items():
        assert 'walk008' not in name.lower()
        if sha(local(name)) != digest:
            raise ValueError('Frozen input changed: ' + name)
    return manifest


def check_qualification(case):
    qual = read(local(case['qualification']))
    total = case['total_controls']
    assert qual['lifecycle_controls'] == total
    assert qual['source_controls'] == case['source_controls']
    assert qual['separate_hold_controls'] == 250
    assert qual['both_quiet_windows_pass'] is True
    assert sha(local(case['trace'])) == qual['traces']['full']['sha256']
    for name, item in qual['independent_reports'].items():
        report = read(local(item['path']))
        assert sha(local(item['path'])) == item['sha256']
        trace_sha = qual['traces']['hold' if name.startswith('hold') else 'full']['sha256']
        assert trace_sha in report.get('input_hashes', report.get('hashes', {})).values()
        if name.endswith('physics'):
            assert report['independent_segment_pass'] is True
            assert report['recorded_trace_reproduced_through_last_sample'] is True
            assert all(report['original_trace_comparison'].values())
        else:
            assert report['requested_segment_quiet_pass'] is True
            assert report['intended_segment_completed'] is True
            assert report['independent_physical_pass'] is True
            if name.startswith('full'):
                assert report['full_lifecycle_source_intent_pass'] is True
                assert report['source_metrics']['source_controls'] == case['source_controls']
    return qual


def validate_trace(case, contract):
    qualification = check_qualification(case)
    trace = archive(local(case['trace']))
    timeline = read(local(case['timeline']))
    phases, total, stop = moving_phases(timeline, case['clip'])
    assert (case['total_controls'], case['moving_stop']) == (total, stop)
    assert trace['qpos'].shape == (total + 1, 30)
    assert trace['qvel'].shape == (total + 1, 29)
    assert trace['target'].shape == (total, 23)
    assert trace[case['history_key']].shape == (total, 300)
    assert trace[case['previous_key']].shape == (total, 23)
    exact(trace['source_frame'], np.arange(11, total + 11, dtype=trace['source_frame'].dtype), 'source frame')
    assert np.all(trace['physics_substeps'] == 10)
    if 'global_control' in trace:
        exact(trace['global_control'], np.arange(total, dtype=trace['global_control'].dtype), 'control clock')
    limits = np.asarray(contract['joint_limits'], np.float64)
    assert np.isfinite(trace['target']).all()
    assert np.all(trace['target'] >= limits[:, 0]) and np.all(trace['target'] <= limits[:, 1])
    elapsed = 0.
    assert trace['physics_expected_time'][0] == trace['physics_time'][0] == elapsed
    for step in range(total * 10):
        elapsed += .002
        assert trace['physics_expected_time'][step + 1] == trace['physics_time'][step + 1] == elapsed
    rows = 0
    digests = {key: hashlib.sha256() for key in ('history', 'previous_action', 'state')}
    boundary = None
    for row in measured_rows(trace, contract, case['history_key'], case['previous_key'], stop):
        rows += 1
        for key, digest in digests.items():
            digest.update(row[key].tobytes())
        if row['control'] == 250:
            boundary = {key: hashlib.sha256(row[key].tobytes()).hexdigest() for key in digests}
    proof = dict(history_rows_byte_exact=rows, all_actual_MPC_initial_prefix=True,
                 source_indices_exact=True, accumulated_native_clock_exact=True,
                 first_selected_control250=boundary,
                 rolling_observation_sha256={key: digest.hexdigest() for key, digest in digests.items()})
    return trace, phases, qualification, proof


def validate_snapshots(case, trace, qualification):
    report = read(local(case['snapshot_report']))
    path = local(case['snapshots'])
    data = archive(path)
    total = case['total_controls']
    assert report['kind'] == 'independent_all_precontrol_native_integration_reconstruction'
    assert report['clip'] == case['clip']
    assert report['full_controls'] == report['precontrol_snapshots'] == total
    assert report['compared_physics_steps'] == total * 10
    assert report['integration_state_size'] == 291
    assert report['all_recorded_samples_byteexact'] is True
    assert report['final291_matches_previously_reconstructed_endpoint_byteexact'] is True
    assert report['independent_physics_already_passed'] is True
    assert report['source_state_rewrites_after_initialization'] == 0
    assert report['new_controller_inference'] is False
    assert report['root_forces'] is False
    assert report['source_features_or_labels_generated'] is False
    assert report['original_trace_sha256'] == qualification['traces']['full']['sha256'] == sha(local(case['trace']))
    assert report['snapshots_sha256'] == sha(path)
    assert report['request_sha256'] == sha(path.parent / 'request.json')
    assert qualification['independent_reports']['full_physics']['sha256'] in report['hashes'].values()
    assert str(data['original_trace_sha256'].item()) == report['original_trace_sha256']
    assert data['integration_spec'].shape == () and int(data['integration_spec']) == 8191
    assert data['control_integration_before'].shape == (total, 291)
    assert data['control_integration_before'].dtype == np.dtype(np.float64)
    assert np.isfinite(data['control_integration_before']).all()
    exact(data['control'], np.arange(total, dtype=np.int64), 'snapshot controls')
    exact(data['qpos'], trace['qpos'][:-1], 'snapshot qpos')
    exact(data['qvel'], trace['qvel'][:-1], 'snapshot qvel')
    exact(data['time'], trace['physics_time'][:-1:10], 'snapshot time')
    exact(data['warning_counts'], trace['physics_warning_number'][:-1:10], 'snapshot warning counts')
    exact(data['warning_lastinfo'], trace['physics_warning_lastinfo'][:-1:10], 'snapshot warning lastinfo')
    exact(data['control_integration_before'][:, 0], data['time'], 'full integration time')
    exact(data['control_integration_before'][:, 1:31], data['qpos'], 'full integration qpos')
    exact(data['control_integration_before'][:, 31:60], data['qvel'], 'full integration qvel')
    assert data['final_integration'].shape == (291,)
    if 'final_integration' in trace:
        exact(data['final_integration'], trace['final_integration'], 'final full integration')
    proof = dict(root_snapshot_report_sha256=sha(local(case['snapshot_report'])),
                 root_snapshots_sha256=sha(path), all_full291_precontrol_rows=total,
                 selected_full291_rows=case['moving_stop'] - 250,
                 qpos_qvel_clock_warning_bytes_exact=True, integration_spec=8191,
                 integration_values_synthesized=False, collector_physics_steps=0)
    return data, proof


def arrays_from_rows(rows):
    result = {}
    count = len(rows['control'])
    for key, shape in SHAPES.items():
        dtype = np.int64 if key in INTEGER else np.float32 if key in FLOAT32 else np.float64
        result[key] = np.asarray(rows[key], dtype=dtype).reshape((count,) + shape)
        assert np.isfinite(result[key]).all()
    return result


def summary(values):
    x = np.asarray(values)
    return dict(minimum=float(np.min(x)), median=float(np.median(x)),
                p95=float(np.percentile(x, 95)), maximum=float(np.max(x)))
