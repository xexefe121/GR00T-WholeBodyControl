"""Independent clipped-feedback, 500Hz physics and original29 intent witness."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
CASE = BASE / 'bfm_online_intent_v2/mpc_h30_323_clip1_v1'
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
from gear_sonic.scripts.evaluate_g1_true23_bfmzero import DATA, MODEL, PHYSICS, load_motion
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


report = json.loads((CASE / 'report.json').read_text())
provenance = json.loads((CASE / 'provenance.json').read_text())
assert sha(CASE / 'trace.npz') == report['trace_sha256']
assert sha(CASE / 'provenance.json') == report['provenance_sha256']
hash_checks = {}
for path, digest in provenance['hashes'].items():
    local = Path(path)
    if not local.is_absolute():
        local = ROOT / local
    if local.name == 'evaluate_g1_true23_mjbatch_plan_replay.py':
        local = CASE / 'runner_snapshot.py'
    hash_checks[str(local)] = sha(local) == digest
assert all(hash_checks.values())
trace = arrays(CASE / 'trace.npz')
plan_dir = Path(provenance['plan'])
plan = arrays(plan_dir / 'trace.npz')
request = json.loads((plan_dir / 'request.json').read_text())
motion, timeline, motion_path = load_motion(report['clip'])
contract = json.loads((BASE / 'mjbatch_native23_inputs_v1/contract.json').read_text())
_, model, physics = prepare_true23_model(ROOT.parent / 'GR00T-WholeBodyControl' / MODEL, ROOT / PHYSICS)
assert (model.nq, model.nv, model.nu) == (30, 29, 23)
assert model.opt.timestep == .002 and report['feedback_correction_clip_rad'] == .1
assert report['failure'] is None and not report['full_source_completed']
assert np.all(trace['physics_substeps'] == 10) and len(trace['target']) == 500
np.testing.assert_array_equal(trace['source_frame'], np.arange(500) + 11)
np.testing.assert_array_equal(trace['qpos'], trace['physics_qpos'][::10])
np.testing.assert_array_equal(trace['qvel'], trace['physics_qvel'][::10])
kp, kd, effort, velocity = (np.asarray(contract[key]) for key in ('kp', 'kd', 'native_effort', 'native_velocity'))
np.testing.assert_array_equal(effort, physics.effort)
limits = model.jnt_range[1:]
np.testing.assert_array_equal(limits, contract['joint_limits'])
raw, predicted = [], []
for control in range(500):
    delta = np.empty(58)
    mujoco.mj_differentiatePos(model, delta[:29], 1., plan['planned_state'][control, :30], trace['qpos'][control])
    delta[29:] = trace['qvel'][control] - plan['planned_state'][control, 30:]
    correction = plan['feedback_gain'][control] @ delta
    raw.append(correction)
    predicted.append(np.clip(plan['planned_target'][control] + np.clip(correction, -.1, .1), limits[:, 0], limits[:, 1]))
raw, predicted = np.asarray(raw), np.asarray(predicted)
np.testing.assert_allclose(raw, trace['raw_feedback_correction'], atol=1e-12, rtol=0)
np.testing.assert_allclose(predicted, trace['target'], atol=1e-12, rtol=0)
active = int(np.sum(np.max(np.abs(raw), axis=1) > .1))
assert active == report['feedback_clip_active_controls'] == 70
expected_torque = np.clip(kp * (np.repeat(predicted, 10, axis=0) - trace['physics_qpos'][:-1, 7:]) -
                          kd * trace['physics_qvel'][:-1, 6:], -effort, effort)
np.testing.assert_allclose(expected_torque, trace['physics_torque'], atol=1e-12, rtol=0)
range_excess = np.maximum(limits[:, 0] - trace['physics_qpos'][1:, 7:], trace['physics_qpos'][1:, 7:] - limits[:, 1]).max(axis=1)
range_by_control = np.maximum(0., range_excess.reshape(500, 10).max(axis=1))
speed_ratio = np.max(np.abs(trace['physics_qvel'][1:, 6:]) / velocity, axis=1)
np.testing.assert_allclose(range_by_control, trace['range_excess'], atol=1e-12, rtol=0)
np.testing.assert_allclose(speed_ratio.reshape(500, 10).max(axis=1), trace['velocity_ratio'], atol=1e-12, rtol=0)
assert range_excess.max() <= 0 and speed_ratio.max() < 1

# Re-execute the physical plant with no state rewrites after initialization.
# Recompute feedback from this independent plant, not saved future states.
plant = mujoco.MjData(model)
plant.qpos[:], plant.qvel[:] = trace['qpos'][0], trace['qvel'][0]
mujoco.mj_forward(model, plant)
qpos_difference = qvel_difference = 0.
for control in range(500):
    delta = np.empty(58)
    mujoco.mj_differentiatePos(model, delta[:29], 1., plan['planned_state'][control, :30], plant.qpos)
    delta[29:] = plant.qvel - plan['planned_state'][control, 30:]
    target = np.clip(plan['planned_target'][control] + np.clip(plan['feedback_gain'][control] @ delta, -.1, .1), limits[:, 0], limits[:, 1])
    for substep in range(10):
        plant.ctrl[:] = np.clip(kp * (target - plant.qpos[7:]) - kd * plant.qvel[6:], -effort, effort)
        mujoco.mj_step(model, plant)
        index = control * 10 + substep + 1
        qpos_difference = max(qpos_difference, float(np.max(np.abs(plant.qpos - trace['physics_qpos'][index]))))
        qvel_difference = max(qvel_difference, float(np.max(np.abs(plant.qvel - trace['physics_qvel'][index]))))
assert qpos_difference < 1e-9 and qvel_difference < 1e-8

# Score with explicit body names and offsets, independently of task_points and
# neutral_wrist_hand_tasks. Original29 source positions are themselves FK checked.
source_model = mujoco.MjModel.from_xml_path(str(ROOT.parent / 'GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_29dof.xml'))
original_path = DATA / 'walk002/original_source_bundle_v1/original_reference.npz'
original = arrays(original_path)
actual_data, source_data = mujoco.MjData(model), mujoco.MjData(source_model)
native_bodies = [model.body(name).id for name in ('left_wrist_roll_rubber_hand', 'right_wrist_roll_rubber_hand', 'torso_link')]
source_bodies = [source_model.body(name).id for name in ('left_wrist_yaw_link', 'right_wrist_yaw_link', 'torso_link')]
native_offsets = np.array(((.264, -.025, 0.), (.264, .025, 0.), (0., 0., .35)))
source_offsets = np.array(((.18, -.025, 0.), (.18, .025, 0.), (0., 0., .35)))
intent_error, source_fk_error = [], 0.
for control, frame in enumerate(trace['source_frame']):
    actual_data.qpos[:] = trace['qpos'][control + 1]
    source_data.qpos[:] = original['source_qpos29'][frame]
    mujoco.mj_kinematics(model, actual_data)
    mujoco.mj_kinematics(source_model, source_data)
    actual = actual_data.xpos[native_bodies] + np.einsum('nij,nj->ni', actual_data.xmat[native_bodies].reshape(3, 3, 3), native_offsets)
    requested = source_data.xpos[source_bodies] + np.einsum('nij,nj->ni', source_data.xmat[source_bodies].reshape(3, 3, 3), source_offsets)
    source_fk_error = max(source_fk_error, float(np.max(np.abs(requested - original['source_task_position_w'][frame]))))
    intent_error.append(np.linalg.norm((actual - actual_data.qpos[:3]) - (requested - source_data.qpos[:3]), axis=1))
intent_error = np.asarray(intent_error)
assert source_fk_error < 1e-12
np.testing.assert_allclose(intent_error, trace['original_relative_hand_head_error'], atol=1e-12, rtol=0)
source_phase = next(phase for phase in timeline['phases'] if phase['name'] == 'source_motion')
selection = slice(source_phase['control_start'], 500)
intent_p95 = np.percentile(intent_error[selection], 95, axis=0)
np.testing.assert_allclose(intent_p95, report['metrics']['original_relative_hand_head_p95_m'], atol=1e-12, rtol=0)

result = dict(kind='independent_H30_clipped_feedback_native323_partial_replay_audit',
              audit_script_sha256=sha(__file__), report_sha256=sha(CASE / 'report.json'),
              trace_sha256=sha(CASE / 'trace.npz'), immutable_input_hash_checks=hash_checks,
              mujoco=mujoco.__version__, feedback_correction_clip_rad=.1,
              control_law='clip_native(u_nominal + clip_each_joint(K @ mj_differentiatePos_and_qvel_error, -0.1, +0.1))',
              feedback_target_reconstruction_max_abs_rad=float(np.max(np.abs(predicted - trace['target']))),
              raw_feedback_reconstruction_max_abs_rad=float(np.max(np.abs(raw - trace['raw_feedback_correction']))),
              feedback_clip_active_controls=active, control_slots=500, physics_steps=5000,
              manual_pd_torque_reconstruction_max_abs_Nm=float(np.max(np.abs(expected_torque - trace['physics_torque']))),
              independent_full_physics_reexecution_qpos_max_abs_difference=qpos_difference,
              independent_full_physics_reexecution_qvel_max_abs_difference=qvel_difference,
              physical_state_rewrites_after_initialization=0, physical_range_excess_max_rad=float(max(0., range_excess.max())),
              physical_velocity_ratio_max=float(speed_ratio.max()),
              commanded_effort_ratio_max=float(np.max(np.abs(trace['physics_torque']) / effort)),
              original_source_task_fk_max_abs_m=source_fk_error,
              independently_scored_original_relative_hand_head_p95_m=intent_p95.tolist(),
              initialization_seconds=7., executed_source_seconds=3., full_source_seconds=13.34,
              full_source_completed=False, full_lifecycle_completed=False,
              planner_p95_ms=report['planning_ms_p50_p95_max'][1],
              conservative_effective_source_preview_seconds=report['conservative_effective_source_preview_seconds'],
              received_stream_controller=False, full_body_tracking_qualified=False, timing_qualified=False,
              hardware_authorized=False)
(CASE / 'independent_audit.json').write_text(json.dumps(result, indent=2, allow_nan=False))
print(json.dumps(result, indent=2, allow_nan=False))
