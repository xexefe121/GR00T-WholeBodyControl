"""Full-source intent and 2ms physical audit; no physical replay or controller edits."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
CASE = Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910/mjbatch_full_v1/walk002_native323_h30_clip1_v1')
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from gear_sonic.scripts.evaluate_g1_true23_bfmzero import DATA, MODEL, PHYSICS, load_motion


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local(path):
    for prefix, drive in (('/mnt/z/', 'Z:/'), ('/mnt/e/', 'E:/'), ('/mnt/c/', 'C:/')):
        if path.startswith(prefix):
            return Path(drive + path[len(prefix):])
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


report = json.loads((CASE / 'report.json').read_text())
request = json.loads((CASE / 'request.json').read_text())
plans = json.loads((CASE / 'plans.json').read_text())
attestation = json.loads((CASE / 'runtime_attestation.json').read_text())
for field, name in (('request_sha256', 'request.json'), ('trace_sha256', 'trace.npz'), ('plans_sha256', 'plans.json')):
    assert sha(CASE / name) == report[field]
assert attestation['request_sha256'] == report['request_sha256']
hash_checks = {}
for path, digest in request['input_hashes'].items():
    path = local(path)
    if path.suffix == '.py':
        path = CASE / (path.stem + '_snapshot.py')
    hash_checks[str(path)] = sha(path) == digest
for path, digest in attestation['mapped_library_hashes'].items():
    hash_checks[str(local(path))] = sha(local(path)) == digest
assert all(hash_checks.values())
assert request['mujoco'] == report['mujoco'] == '3.2.3'
assert request.get('motion_override') is None
assert (request['source_clock_hz'], request['physics_hz'], request['source_reference_lift_m'], request['source_frame_removal']) == (50, 500, 0., 0)
trace = read_npz(CASE / 'trace.npz')
motion, timeline, motion_path = load_motion('walk002')
original_path = DATA / 'walk002/original_source_bundle_v1/original_reference.npz'
original = read_npz(original_path)
contract = json.loads((BASE / 'mjbatch_native23_inputs_v1/contract.json').read_text())
manifest = json.loads((BASE / 'mjbatch_native23_inputs_v1/manifest.json').read_text())
assert manifest == request['model_manifest']
assert sha(motion_path) == manifest['cases']['walk002']['native_original.npz']
assert sha(original_path) == manifest['cases']['walk002']['original29.npz']
model_path = ROOT.parent / 'GR00T-WholeBodyControl' / MODEL
source_model_path = ROOT.parent / 'GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_29dof.xml'
assert sha(model_path) == manifest['prepared_source_model_sha256']
assert sha(ROOT / PHYSICS) == manifest['physics_sha256']
native, source = mujoco.MjModel.from_xml_path(str(model_path)), mujoco.MjModel.from_xml_path(str(source_model_path))
limits = np.asarray(contract['joint_limits'])
np.testing.assert_array_equal(native.jnt_range[1:], limits)
kp, kd, effort, velocity = (np.asarray(contract[key]) for key in ('kp', 'kd', 'native_effort', 'native_velocity'))
n = len(trace['target'])
assert n == 1417 == timeline['total_requested_controls'] == report['completed_controls']
assert report['failure'] is None and np.all(trace['physics_substeps'] == 10)
np.testing.assert_array_equal(trace['source_frame'], np.arange(n) + 11)
np.testing.assert_array_equal(trace['qpos'], trace['physics_qpos'][::10])
np.testing.assert_array_equal(trace['qvel'], trace['physics_qvel'][::10])
assert len(trace['physics_qpos']) == 14171 and len(trace['physics_torque']) == 14170
expected_torque = np.clip(kp * (np.repeat(trace['target'], 10, axis=0) - trace['physics_qpos'][:-1, 7:]) - kd * trace['physics_qvel'][:-1, 6:], -effort, effort)
np.testing.assert_allclose(expected_torque, trace['physics_torque'], atol=1e-10, rtol=0)
range_by_joint = np.maximum(limits[:, 0] - trace['physics_qpos'][:, 7:], trace['physics_qpos'][:, 7:] - limits[:, 1])
range_by_step = np.maximum(0., range_by_joint.max(axis=1))
speed_by_step = np.max(np.abs(trace['physics_qvel'][:, 6:]) / velocity, axis=1)
effort_by_step = np.max(np.abs(trace['physics_torque']) / effort, axis=1)
np.testing.assert_allclose(range_by_step[1:].reshape(n, 10).max(axis=1), trace['range_excess'], atol=1e-12, rtol=0)
np.testing.assert_allclose(speed_by_step[1:].reshape(n, 10).max(axis=1), trace['velocity_ratio'], atol=1e-12, rtol=0)
assert np.max(speed_by_step) <= 1 and np.max(effort_by_step) <= 1 + 1e-12
violating_steps = np.flatnonzero(range_by_step > 0)
worst_step, worst_joint = (int(value) for value in np.unravel_index(np.argmax(range_by_joint), range_by_joint.shape))
blocks = []
for block in np.split(violating_steps, np.flatnonzero(np.diff(violating_steps) > 1) + 1):
    blocks.append(dict(first_step=int(block[0]), last_step=int(block[-1]), samples=len(block),
                       first_time_s=float(block[0] * .002), last_time_s=float(block[-1] * .002)))

actual_data, source_data = mujoco.MjData(native), mujoco.MjData(source)
native_names = ('left_wrist_roll_rubber_hand', 'right_wrist_roll_rubber_hand', 'torso_link', 'left_ankle_roll_link', 'right_ankle_roll_link')
source_names = ('left_wrist_yaw_link', 'right_wrist_yaw_link', 'torso_link', 'left_ankle_roll_link', 'right_ankle_roll_link')
native_ids = [native.body(name).id for name in native_names]
source_ids = [source.body(name).id for name in source_names]
native_offsets = np.array(((.264, -.025, 0), (.264, .025, 0), (0, 0, .35), (0, 0, 0), (0, 0, 0)))
source_offsets = np.array(((.18, -.025, 0), (.18, .025, 0), (0, 0, .35), (0, 0, 0), (0, 0, 0)))
values = {key: [] for key in ('original_world_task_error', 'original_root_relative_task_error', 'original_foot_orientation_error_rad', 'original_hand_orientation_error_rad', 'original_torso_orientation_error_rad', 'original_root_yaw_error_deg', 'native_world_foot_error', 'native_root_relative_foot_error')}
source_fk_error = 0.
for control, frame in enumerate(trace['source_frame']):
    actual_data.qpos[:] = trace['qpos'][control + 1]
    source_data.qpos[:] = original['source_qpos29'][frame]
    mujoco.mj_kinematics(native, actual_data)
    mujoco.mj_kinematics(source, source_data)
    actual_rot = actual_data.xmat[native_ids].reshape(5, 3, 3)
    original_rot = source_data.xmat[source_ids].reshape(5, 3, 3)
    actual_tasks = actual_data.xpos[native_ids] + np.einsum('nij,nj->ni', actual_rot, native_offsets)
    original_tasks = source_data.xpos[source_ids] + np.einsum('nij,nj->ni', original_rot, source_offsets)
    source_fk_error = max(source_fk_error, float(np.max(np.abs(original_tasks[:3] - original['source_task_position_w'][frame]))))
    values['original_world_task_error'].append(np.linalg.norm(actual_tasks - original_tasks, axis=1))
    values['original_root_relative_task_error'].append(np.linalg.norm((actual_tasks - actual_data.qpos[:3]) - (original_tasks - source_data.qpos[:3]), axis=1))
    orientation_error = Rotation.from_matrix(original_rot.transpose(0, 2, 1) @ actual_rot).magnitude()
    values['original_foot_orientation_error_rad'].append(orientation_error[3:])
    values['original_hand_orientation_error_rad'].append(orientation_error[:2])
    values['original_torso_orientation_error_rad'].append(orientation_error[2])
    actual_root_rot = actual_data.xmat[native.body('pelvis').id].reshape(3, 3)
    original_root_rot = source_data.xmat[source.body('pelvis').id].reshape(3, 3)
    dyaw = np.arctan2(actual_root_rot[1, 0], actual_root_rot[0, 0]) - np.arctan2(original_root_rot[1, 0], original_root_rot[0, 0])
    values['original_root_yaw_error_deg'].append(abs(np.degrees(np.arctan2(np.sin(dyaw), np.cos(dyaw)))))
    desired_native_feet = motion['body_pos_w'][frame, np.asarray(native_ids[3:]) - 1]
    world_foot_delta = actual_tasks[3:] - desired_native_feet
    root_delta = actual_data.qpos[:3] - motion['body_pos_w'][frame, 0]
    values['native_world_foot_error'].append(np.linalg.norm(world_foot_delta, axis=1))
    values['native_root_relative_foot_error'].append(np.linalg.norm(world_foot_delta - root_delta, axis=1))
values = {key: np.asarray(value) for key, value in values.items()}
assert source_fk_error < 1e-12
source_phase = next(phase for phase in timeline['phases'] if phase['name'] == 'source_motion')
assert (source_phase['control_start'], source_phase['control_stop'], source_phase['requested_controls']) == (350, 1017, 667)
source_slice = slice(350, 1017)
np.testing.assert_allclose(np.percentile(values['native_world_foot_error'][source_slice], 95, axis=0), report['metrics']['tracked_body_position_p95'][2:4], atol=1e-10, rtol=0)
phase_metrics = []
for phase in timeline['phases']:
    selection = slice(phase['control_start'], phase['control_stop'])
    row = dict(name=phase['name'], controls=phase['requested_controls'], full_controls=phase['requested_controls'],
        start_seconds=phase['control_start']*.02, stop_seconds=phase['control_stop']*.02,
        source_frame_start=phase['frame_start'], source_frame_stop_exclusive=phase['frame_stop'],
        leg_rmse=float(np.sqrt(np.mean(trace['joint_error'][selection, :12]**2))),
        root_p95_m=float(np.percentile(np.linalg.norm(trace['root_error'][selection], axis=1), 95)))
    row.update({key + '_p95': np.percentile(value[selection], 95, axis=0).tolist() for key, value in values.items()})
    phase_metrics.append(row)
source_metrics = next(row for row in phase_metrics if row['name'] == 'source_motion')
last = trace['qpos'][-1]
final_joint_velocity_max = float(np.max(np.abs(trace['qvel'][-1, 6:])))
control = (worst_step - 1) // 10
target = float(trace['target'][control, worst_joint])
overshoot = float(range_by_step[worst_step])
current_margin = request['costs']['joint_limit_margin']
current_weight = request['costs']['near_joint_limit']
penalty_grid = [dict(margin_rad=margin, weight=weight, local_peak_state_penalty=weight * (margin + overshoot)**2,
                     local_peak_inward_penalty_gradient=2 * weight * (margin + overshoot))
                for margin, weight in ((current_margin, current_weight), (.02, 200.), (.03, 200.), (.01, 1000.), (.02, 1000.), (.03, 1000.))]
result = dict(kind='independent_full_native323_walk002_intent_physical_audit',
    audit_script_sha256=sha(__file__), report_sha256=sha(CASE/'report.json'), trace_sha256=sha(CASE/'trace.npz'),
    runtime_attestation_sha256=sha(CASE/'runtime_attestation.json'), immutable_input_hash_checks=hash_checks,
    original_native_reference_sha256=sha(motion_path), original29_reference_sha256=sha(original_path),
    native_model_sha256=sha(model_path), original29_model_sha256=sha(source_model_path),
    physical_replay_performed=False, source_frames_removed=0, source_timing_scale=1.,
    full_lifecycle_controls=1417, full_source_controls=667, all_controls_have_10_substeps=True,
    physical_steps=14170, simulated_seconds=28.34, source_control_slice=[350,1017],
    source_pose_frame_slice=[361,1028], original29_source_task_fk_max_abs_m=source_fk_error,
    task_order=['left_hand','right_hand','head_proxy','left_ankle_origin','right_ankle_origin'],
    relative_metric_convention='Subtract each robot root translation separately; preserve world axes. No yaw alignment or fitted transform.',
    source_metrics=source_metrics, phase_metrics=phase_metrics,
    foot_metric_discrepancy_resolved='Producer tracked_body_position is absolute world ankle-origin error. Referee relative_landmark subtracts actual-minus-reference root translation before taking the same world-axis ankle-origin norm. Neither changes ankle landmark or axes.',
    strict_joint_range_pass=False, strict_range_excess_max_rad=overshoot, strict_range_violation_samples=len(violating_steps),
    strict_range_violation_blocks=blocks, velocity_limits_pass=True, velocity_ratio_max=float(speed_by_step.max()),
    effort_limits_pass=True, effort_ratio_max=float(effort_by_step.max()),
    pd_torque_reconstruction_max_abs_Nm=float(np.max(np.abs(expected_torque-trace['physics_torque']))),
    worst_joint=dict(name=contract['joint_names'][worst_joint], hardware_index=worst_joint, physical_step=worst_step,
        time_s=worst_step*.002, control_index=control, active_source_frame=int(trace['source_frame'][control]),
        q_rad=float(trace['physics_qpos'][worst_step,7+worst_joint]), limits_rad=limits[worst_joint].tolist(),
        dq_radps=float(trace['physics_qvel'][worst_step,6+worst_joint]), pd_target_rad=target,
        source_joint_goal_rad=float(motion['joint_pos'][trace['source_frame'][control],worst_joint])),
    final_root_error_m=float(np.linalg.norm(trace['root_error'][-1])), final_joint_velocity_max_radps=final_joint_velocity_max,
    final_root_height_m=float(last[2]), final_root_tilt_deg=float(np.degrees(np.arccos(np.clip(1-2*np.sum(last[4:6]**2),-1,1)))),
    planning_ms_p50_p95_max=report['planning_ms_p50_p95_max'], planning_deadlines_missed=report['planning_deadlines_missed'],
    plans=len(plans), planning_commit_budget_ms=100, conservative_effective_source_preview_seconds=.74,
    joint_margin_analysis=dict(current_margin_rad=current_margin,current_weight=current_weight,
        event_specific_target_inset_lower_bound_rad=overshoot,
        inset_assumption='Only if the same outward tracking undershoot persisted after shifting the saturated target inward; this is not a controller guarantee.',
        penalty_grid_at_observed_peak=penalty_grid,
        minimum_safe_weight_identifiable_from_single_trace=False,
        reason='A soft penalty cannot guarantee a strict state bound. The nominal planned trajectory itself crosses the ankle limit; tuning weight/margin requires new closed-loop optimization and every-2ms validation. No unique minimum weight follows from this trajectory.',
        suggested_bounded_followup='Test margin0.02 or0.03 rad with weight1000 as explicit ablations; preserve all goals and timing. Require a strict physics-feasibility gate before accepting a teacher.'),
    received_stream_controller=False, full_body_tracking_qualified=False, timing_qualified=False, hardware_authorized=False)
np.savez_compressed(CASE/'independent_intent_metrics.npz', **values, strict_range_excess_by_physics_step=range_by_step)
(CASE/'independent_full_lifecycle_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False))
print(json.dumps({key:result[key] for key in ('full_source_controls','source_metrics','strict_range_excess_max_rad','strict_range_violation_blocks','worst_joint','final_root_error_m','final_joint_velocity_max_radps')},indent=2))
