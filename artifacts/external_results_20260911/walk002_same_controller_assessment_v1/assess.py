"""Read-only recorded-run inventory and one proposed command; no simulator import."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
OLD = Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910')
ROOT = Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
BUNDLE = ROOT / 'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'
roots = [NEW, OLD, ROOT / 'artifacts']


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


files = subprocess.check_output(['rg', '--files', *map(str, roots), '-g', 'request.json', '-g', 'report.json'], text=True).splitlines()
walk_requests, walk_reports, parse_errors = [], [], []
for name in files:
    path = Path(name)
    try:
        item = json.loads(path.read_text(encoding='utf-8-sig'))
    except (ValueError, OSError) as error:
        parse_errors.append(dict(path=name, error=str(error)))
        continue
    if item.get('clip') != 'walk002':
        continue
    (walk_requests if path.name == 'request.json' else walk_reports).append((path, item))

qualified_path = NEW / 'pico_full_control_lm_v1/request.json'
qualified = json.loads(qualified_path.read_text())
config_keys = ['horizon', 'iterations', 'commit', 'batch_threads', 'finite_difference_epsilon',
               'feedback_correction_clip_rad', 'costs', 'hard_feasibility', 'restoration']
comparisons = []
for path, request in walk_requests:
    if request.get('kind') != 'offline_native23_mjbatch_ilqr_probe':
        continue
    report_path = path.with_name('report.json')
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    delta = {key: dict(existing=request.get(key), qualified=qualified.get(key))
             for key in config_keys if request.get(key) != qualified.get(key)}
    fresh = request.get('fresh_bfm_seed')
    seed_delta = dict(fresh_bfm_present=bool(fresh), recorded_bfm_present=bool(request.get('recorded_target_seed')),
                      original_fresh_goal_horizon=fresh.get('original_goal_horizon') if fresh else None,
                      original_fresh_position_gain=fresh.get('position_gain') if fresh else None,
                      original_fresh_yaw_gain=fresh.get('yaw_gain') if fresh else None)
    full_match = not delta and bool(fresh) and bool(request.get('recorded_target_seed'))
    comparisons.append(dict(request=str(path), request_sha256=sha(path),
                            report_sha256=sha(report_path) if report_path.exists() else None,
                            probe=request.get('probe'), requested_controls=request.get('requested_controls'),
                            completed_controls=report.get('completed_controls'), failure=report.get('failure'),
                            source_reference=request.get('motion_override'), mujoco=request.get('mujoco'),
                            same_qualified_solver_config=full_match, config_differences=delta, seed_settings=seed_delta))

failed_requests = [row for row in comparisons if row['failure'] and row['failure'].get('time') is not None]
earliest_producer_failure = min(failed_requests, key=lambda row: row['failure']['time']) if failed_requests else None

# Scan saved native arrays only. No manual replay, forward, optimizer, or inference.
with np.load(BUNDLE / 'prepared_model_arrays.npz', allow_pickle=False) as arrays:
    range_key = next(key for key in arrays.files if key.split('/')[-1] == 'jnt_range')
    limits = arrays[range_key][1:]
contract = json.loads((BUNDLE / 'contract.json').read_text())
timeline = json.loads((BUNDLE / 'walk002/timeline.json').read_text())
source_start = next(phase['control_start'] for phase in timeline['phases'] if phase['name'] == 'source_motion')
full_cases = []
for entry in comparisons:
    if entry['probe'] != 'full-lifecycle':
        continue
    path = Path(entry['request']).with_name('trace.npz')
    if not path.exists():
        continue
    with np.load(path, allow_pickle=False) as arrays:
        if 'physics_qpos' not in arrays:
            continue
        qpos, qvel = arrays['physics_qpos'], arrays['physics_qvel']
        excess = np.maximum(limits[:, 0] - qpos[:, 7:], qpos[:, 7:] - limits[:, 1])
        violations = np.argwhere(excess > 1e-6)
        first = None
        if len(violations):
            sample, joint = map(int, violations[0])
            first = dict(physics_sample=sample, actual_control_zero_based=(sample - 1) // 10,
                         control_slot_one_based=(sample - 1) // 10 + 1, substep=(sample - 1) % 10 + 1,
                         simulation_seconds=sample * .002, source_seconds=sample * .002 - source_start * .02,
                         native_joint_index=joint, qpos=float(qpos[sample, joint + 7]),
                         qvel=float(qvel[sample, joint + 6]), excess_rad=float(excess[sample, joint]),
                         native_bounds=limits[joint].tolist())
        full_cases.append(dict(request=entry['request'], trace=str(path), trace_sha256=sha(path),
                               saved_physics_steps=len(qpos) - 1, first_strict_range_violation=first,
                               strict_range_peak=float(max(0, excess.max())),
                               speed_ratio_peak=float(np.max(np.abs(qvel[:, 6:]) / contract['native_velocity'])),
                               producer_failure=entry['failure']))

v4 = OLD / 'mjbatch_full_v1/walk002_v4_native323_full_v1'
reference_dir = OLD / 'mjbatch_intent_floor_inputs_v1/walk002'
seed = ROOT / 'artifacts/teleop_six_hour_20260910/bfm_walk002_feedback_v2'
frozen = NEW / 'pico_control_lm_integration_v1/repo'
assets = [qualified_path, v4 / 'request.json', v4 / 'report.json', v4 / 'trace.npz',
          v4 / 'g1_true23_mjbatch_ilqr_core_snapshot.py',
          OLD / 'mjbatch_full_v1/walk002_v4_native323_rawlimits_v1.json',
          NEW / 'initial_seed_preflight_v1/walk002/report.json',
          BUNDLE / 'manifest.json', BUNDLE / 'contract.json', BUNDLE / 'native_prepared.xml',
          BUNDLE / 'prepared_model_arrays.npz', BUNDLE / 'walk002/native_original.npz',
          BUNDLE / 'walk002/original29.npz', BUNDLE / 'walk002/timeline.json',
          seed / 'trace.npz', seed / 'report.json',
          ROOT / 'artifacts/teleop_six_hour_20260910/bfm_onnx_v2/actor.onnx',
          ROOT / 'artifacts/teleop_six_hour_20260910/bfm_onnx_v2/backward.onnx',
          ROOT / 'artifacts/teleop_six_hour_20260910/bfm_onnx_v2/manifest.json',
          NEW / 'pico_control_lm_integration_v1/frozen_source_hashes.json']
assets += list(frozen.rglob('*.py'))
assets += [path for path in reference_dir.iterdir() if path.is_file()]
hashes = {str(path): sha(path) for path in assets}
receipt = json.loads((reference_dir / 'portable_receipt.json').read_text())
assert hashes[str(reference_dir / 'reference.npz')] == receipt['reference_sha256']
assert hashes[str(BUNDLE / 'walk002/native_original.npz')] == receipt['base_native_bundle_reference_sha256']
seed_report = json.loads((seed / 'report.json').read_text())
assert seed_report['clip'] == 'walk002' and seed_report['failure'] is None
with np.load(reference_dir / 'reference.npz') as arrays:
    frame_count = len(arrays['joint_pos'])
    assert frame_count == timeline['total_frames'] == 1428
with np.load(seed / 'trace.npz') as arrays:
    assert arrays['target'].shape == (1417, 23)
    assert np.isfinite(arrays['target']).all()

proposal = (NEW / 'launch_pico_full_control_lm_v1.ps1').read_text()
proposal = proposal.replace("'--clip', 'pico'", "'--clip', 'walk002'")
proposal = proposal.replace('bfm_pico_feedback_v2', 'bfm_walk002_feedback_v2')
proposal = proposal.replace('mjbatch_intent_floor_inputs_v1/pico/reference.npz', 'mjbatch_intent_floor_inputs_v1/walk002/reference.npz')
proposal = proposal.replace('pico_full_control_lm_v1', 'walk002_full_control_lm_v1')
proposal_path = HERE / 'proposed_launch.ps1.txt'
proposal_path.write_text(proposal)

assessment = dict(utc=datetime.now(timezone.utc).isoformat(), scan_roots=list(map(str, roots)),
                  scanned_request_report_files=len(files), parse_errors=parse_errors,
                  walk002_requests=len(walk_requests), walk002_reports=len(walk_reports),
                  walk002_mpc_runs=comparisons, saved_full_trace_limits=full_cases,
                  earliest_mpc_producer_abort_by_simulation_time=earliest_producer_failure,
                  exact_qualified_configuration_ever_recorded=any(row['same_qualified_solver_config'] for row in comparisons),
                  scope_limit='Absence established in the three listed local artifact roots; no claim about unrecorded external runs.',
                  initial_seed_preflight='Existing NEW initial_seed_preflight_v1 only certifies three initialH30 candidates, no actual full MPC lifecycle.',
                  old_v4_epsilon_note='Old request lacks finite_difference_epsilon field, but frozen core snapshot line10 declares EPS=1e-6; numerical epsilon is unchanged, request provenance is newer.',
                  one_proposal=dict(kind='one fresh same-controller full-lifecycle walk002',
                                    requested_controls=1417, source_controls=667, source_seconds=13.34,
                                    lifecycle_seconds=28.34, requested_native_steps=14170,
                                    canonical_initialization='new native MjData at unchanged v4 frame10; original native reference for both BFM seeds',
                                    source_control_interval=[350, 1017], terminal_standing_interval=[1117, 1417],
                                    source_frame_interval=[361, 1028], no_frame_removal=True,
                                    evaluator=str(frozen / 'gear_sonic/scripts/evaluate_g1_true23_mjbatch_mpc.py'),
                                    configuration='same PICO H30/5/commit5/8threads/EPS1e-6/feedback.1/allmargin.05weight2000/relativefoot400/hard/native+guided/K0/thirdLM; original recorded+freshBFM8pos1yaw2',
                                    command_file=str(proposal_path), command_sha256=sha(proposal_path),
                                    output=str(NEW / 'walk002_full_control_lm_v1'),
                                    output_absent=not (NEW / 'walk002_full_control_lm_v1').exists(),
                                    differences_from_pico=['clip', 'recorded clip-specific BFM seed', 'clip-specific unchanged v4 reference', 'output/log filenames', 'timeline-derived1417/667 counts'],
                                    solver_or_reference_changes=False, run_started=False,
                                    gates='Retain every2ms strict gates; stop safely on rejection; independent native all14170 replay and original667-source/last3s quiet inspection required before qualification.',
                                    timing='Offline evidence only; no realtime or hardware inference. Separate250 continuation only after full result review.'),
                  source_asset_hashes=hashes, assessment_script_sha256=sha(Path(__file__)),
                  new_physics=False, optimizer=False, policy_inference=False, solver_edits=False, reference_edits=False)
(HERE / 'assessment.json').write_text(json.dumps(assessment, indent=2))
summary = dict(scanned=len(files), parse_errors=len(parse_errors), mpc_runs=len(comparisons),
               exact_qualified_configuration_ever_recorded=assessment['exact_qualified_configuration_ever_recorded'],
               full_cases=[dict(name=Path(row['trace']).parent.name, first=row['first_strict_range_violation'], failure=row['producer_failure']) for row in full_cases],
               assessment_sha256=sha(HERE / 'assessment.json'), command_sha256=sha(proposal_path), assets=len(hashes))
print(json.dumps(summary, indent=2))
