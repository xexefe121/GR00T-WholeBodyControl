"""Read-only request/atomic-progress monitor for the one canonical walk002 run."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
OUTPUT = BASE / 'walk002_full_control_lm_v1'
ROOT = Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


status = dict(utc=datetime.now(timezone.utc).isoformat())
if not (OUTPUT / 'request.json').exists():
    status['stage'] = 'initialization_before_request'
    print(json.dumps(status))
    raise SystemExit(0)
canonical_path = BASE / 'pico_full_control_lm_v1/request.json'
request_path = OUTPUT / 'request.json'
canonical = json.loads(canonical_path.read_text())
request = json.loads(request_path.read_text())
old_v4 = json.loads(Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910/mjbatch_full_v1/walk002_v4_native323_full_v1/request.json').read_text())
timeline = json.loads((ROOT / 'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/walk002/timeline.json').read_text())
phase = next(item for item in timeline['phases'] if item['name'] == 'source_motion')
common = ['probe', 'horizon', 'commit', 'iterations', 'finite_difference_epsilon', 'batch_threads',
          'feedback_correction_clip_rad', 'costs', 'mujoco', 'restoration', 'hard_feasibility',
          'source_clock_hz', 'physics_hz', 'source_reference_lift_m', 'source_frame_removal',
          'actual_execution', 'planning_actuators', 'feedback_gain_semantics', 'exact_solver_reuse',
          'physical_initialization', 'declared_future_preview_seconds', 'conservative_effective_source_preview_seconds',
          'conservative_effective_raw_pose_support_seconds']
checks = {key: canonical[key] == request[key] for key in common}
checks.update(clip=request['clip'] == 'walk002', requested=request['requested_controls'] == 1417 == timeline['total_requested_controls'],
              source_requested=request['source_requested_controls'] == 667,
              exact_source_phase=request['source_phase'] == phase,
              canonical_initial_qpos=request['initial_qpos'] == old_v4['initial_qpos'],
              canonical_initial_qvel=request['initial_qvel'] == old_v4['initial_qvel'],
              v4_reference=request['motion_override']['reference_sha256'] == '9486ae6a58942cbe636f70a90e8006fbbec98d228519bda7a59f0e0331c6c562',
              recorded_seed=request['recorded_target_seed']['trace_sha256'] == '74bbd9c8e4cae7c8c8b421b4a1448f917872161b837efce04001810041b94f97',
              recorded_seed_count=request['recorded_target_seed']['controls'] == 1417,
              fresh_original_reference=request['fresh_bfm_seed']['original_reference_sha256'] == '94cbd249255716cfd4cfe6eac290b8e0767a3efb262c1bde22402b971a729990')
fresh_exclusions = {'original_goal_arrays_sha256', 'original_reference_sha256'}
checks['same_fresh_bfm_configuration'] = ({key: value for key, value in request['fresh_bfm_seed'].items() if key not in fresh_exclusions}
                                         == {key: value for key, value in canonical['fresh_bfm_seed'].items() if key not in fresh_exclusions})
recorded_exclusions = {'path', 'trace_sha256', 'report_sha256', 'controls'}
checks['same_recorded_seed_configuration'] = ({key: value for key, value in request['recorded_target_seed'].items() if key not in recorded_exclusions}
                                            == {key: value for key, value in canonical['recorded_target_seed'].items() if key not in recorded_exclusions})
if not all(checks.values()):
    raise ValueError('actual request mismatch: ' + repr({key: value for key, value in checks.items() if not value}))
(HERE / 'actual_request_verification.json').write_text(json.dumps(dict(all_exact=True, count=len(checks), checks=checks,
    request_sha256=sha(request_path), qualified_request_sha256=sha(canonical_path)), indent=2))
status.update(request_checks_pass=len(checks), requested_controls=1417, source_requested_controls=667)
if (OUTPUT / 'report.json').exists():
    report = json.loads((OUTPUT / 'report.json').read_text())
    status.update(stage='final', **{key: report[key] for key in ('completed_controls', 'physics_steps', 'failure', 'probe_completed',
                                                               'restoration_triggers', 'restoration_control_lm_retries')})
    status['trace_sha256'] = sha(OUTPUT / 'trace.npz')
    status['report_sha256'] = sha(OUTPUT / 'report.json')
    receipt = json.loads((HERE / 'launch_receipt.json').read_text())
    exact = {path: sha(Path(path)) == expected for path, expected in receipt['verified_hashes'].items()}
    assert all(exact.values()), exact
    (HERE / 'postrun_provenance.json').write_text(json.dumps(dict(all_exact=True, count=len(exact), checks=exact), indent=2))
    status['launch_hashes_unchanged'] = len(exact)
elif (OUTPUT / 'trace.partial.npz').exists():
    path = OUTPUT / 'trace.partial.npz'
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['checkpoint_metadata']))
    if metadata['requested_controls'] != 1417 or metadata['requested_source_controls'] != 667:
        raise ValueError('checkpoint truncated intended request')
    status.update(stage='incomplete_checkpoint', **{key: metadata[key] for key in ('completed_controls', 'completed_full_controls',
        'partial_controls', 'completed_source_controls', 'physics_steps', 'simulation_time', 'failure')})
    # Preserve exact checkpoint bytes, then read its own metadata to avoid races with atomic replacement.
    if metadata['completed_controls'] in (400, 500, 1400):
        destination = HERE / ('checkpoint_%04d.npz' % metadata['completed_controls'])
        if not destination.exists():
            shutil.copyfile(path, destination)
            with np.load(destination, allow_pickle=False) as archive:
                captured = json.loads(str(archive['checkpoint_metadata']))
            (destination.with_suffix('.json')).write_text(json.dumps(dict(
                actual_completed_controls=captured['completed_controls'], sha256=sha(destination),
                saved_arrays_only=True, nominal_filename_control=metadata['completed_controls']), indent=2))
else:
    status['stage'] = 'before_first_checkpoint'
if (OUTPUT / 'restoration_events.json').exists():
    events = json.loads((OUTPUT / 'restoration_events.json').read_text())
    status['restoration_events'] = [dict(control=event['control'], accepted=event['accepted'], modes=[attempt['mode'] for attempt in event['attempts']]) for event in events]
with (HERE / 'monitor.jsonl').open('a') as stream:
    stream.write(json.dumps(status) + '\n')
print(json.dumps(status))
