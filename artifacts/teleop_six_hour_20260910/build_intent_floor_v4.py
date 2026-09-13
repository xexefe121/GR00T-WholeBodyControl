"""Explicit upward reference adaptation; original source and physical floor stay unchanged."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910')
INPUT = ARCHIVE / 'mjbatch_intent_inputs_v1'
RAW = ARCHIVE / 'intent_retarget_v4_floor_v1'
PORTABLE = ARCHIVE / 'mjbatch_intent_floor_inputs_v1'
AUTHORITY = ROOT / 'gear_sonic/scripts/evaluate_g1_true23_bfmzero.py'
AUTHORITY_BYTES = AUTHORITY.read_bytes()
OMEGA, BUFFER, CLEARANCE_EPS, DT = 15., .020, .000001, .020
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from gear_sonic.scripts.evaluate_g1_true23_bfmzero import DATA, MODEL, PHYSICS, load_case_motion


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def copy_exact(source, destination):
    assert not destination.exists()
    shutil.copy2(source, destination)
    assert sha(source) == sha(destination)


def wsl(path):
    text = str(Path(path).absolute()).replace('\\', '/')
    return '/mnt/' + text[0].lower() + '/' + text[3:]


def lift_profile(raw):
    """Fixed-parameter zero-lookahead pose filter plus explicit clearance guard."""
    smooth = np.empty_like(raw)
    smooth[0], velocity = raw[0], 0.
    decay = np.exp(-OMEGA * DT)
    for i in range(1, len(raw)):
        offset = smooth[i - 1] - raw[i]
        combination = velocity + OMEGA * offset
        smooth[i] = raw[i] + (offset + combination * DT) * decay
        velocity = (velocity - OMEGA * combination * DT) * decay
    candidate = smooth + BUFFER
    guard = raw + CLEARANCE_EPS > candidate
    return np.maximum(raw + CLEARANCE_EPS, candidate), smooth, guard


for directory in (RAW, PORTABLE):
    assert directory.resolve().is_relative_to(ARCHIVE.resolve())
    directory.mkdir(exist_ok=False)
    (directory / 'floor_transform_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (directory / 'validation_authority_snapshot.py').write_bytes(AUTHORITY_BYTES)
native_path = ROOT.parent / 'GR00T-WholeBodyControl' / MODEL
native = mujoco.MjModel.from_xml_path(str(native_path))
floor = native.geom('floor').id
feet = [native.body(side + '_ankle_roll_link').id for side in ('left', 'right')]
spheres = [[i for i in range(native.ngeom) if native.geom_bodyid[i] == body and
            native.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE and native.geom_contype[i] != 0] for body in feet]
assert spheres == [[15, 16, 17, 18], [30, 31, 32, 33]]
assert native.geom_type[floor] == mujoco.mjtGeom.mjGEOM_PLANE
data = mujoco.MjData(native)


def foot_clearance(motion):
    result = np.empty((len(motion['joint_pos']), 2))
    for frame in range(len(result)):
        data.qpos[:] = np.r_[motion['body_pos_w'][frame, 0], motion['body_quat_w'][frame, 0], motion['joint_pos'][frame]]
        mujoco.mj_kinematics(native, data)
        normal = data.geom_xmat[floor].reshape(3, 3)[:, 2]
        np.testing.assert_array_equal(normal, [0, 0, 1])
        for side, ids in enumerate(spheres):
            result[frame, side] = np.min((data.geom_xpos[ids] - data.geom_xpos[floor]) @ normal - native.geom_size[ids, 0])
    return result


entries = []
for clip in ('pico', 'walk003', 'walk008', 'walk002'):
    source, output, portable = INPUT / clip, RAW / clip, PORTABLE / clip
    output.mkdir(); portable.mkdir()
    source_receipt = json.loads((source / 'portable_receipt.json').read_text())
    assert sha(source / 'reference.npz') == source_receipt['reference_sha256']
    before = read_npz(source / 'reference.npz')
    before_clearance = foot_clearance(before)
    raw_required = np.maximum(0., -before_clearance.min(axis=1))
    lift, smooth, guard = lift_profile(raw_required)
    after = {key: value.copy() for key, value in before.items()}
    after['body_pos_w'][:, :, 2] += lift[:, None]
    after['body_lin_vel_w'][:, :, 2] = np.gradient(after['body_pos_w'][:, :, 2], DT, axis=0)
    after_clearance = foot_clearance(after)
    np.testing.assert_allclose(after_clearance, before_clearance + lift[:, None], atol=2e-10, rtol=0)
    assert after_clearance.min() >= -1e-10
    for end in (11, 361, min(1000, len(lift)), len(lift) - 1):
        np.testing.assert_array_equal(lift_profile(raw_required[:end])[0], lift[:end])
    for key in before:
        if key not in ('body_pos_w', 'body_lin_vel_w'):
            np.testing.assert_array_equal(before[key], after[key])
    np.testing.assert_array_equal(before['body_pos_w'][:, :, :2], after['body_pos_w'][:, :, :2])
    np.testing.assert_array_equal(before['body_lin_vel_w'][:, :, :2], after['body_lin_vel_w'][:, :, :2])
    for name in ('original_timeline.json', 'retarget_runner_snapshot.py', 'arm_ik_snapshot.py', 'hand_geometry_snapshot.py', 'physics_contract_snapshot.json'):
        copy_exact(source / name, output / name)
    copy_exact(source / 'reference.npz', output / 'before_floor_reference.npz')
    copy_exact(source / 'report.json', output / 'before_floor_report.json')
    copy_exact(source / 'portable_receipt.json', output / 'before_floor_portable_receipt.json')
    np.savez_compressed(output / 'reference.npz', **after)
    np.savez_compressed(output / 'frame_lift.npz', frame_lift_m=lift, raw_required_lift_m=raw_required,
        before_foot_clearance_m=before_clearance, after_foot_clearance_m=after_clearance,
        filtered_required_lift_m=smooth, clearance_guard_active=guard)
    original = read_npz(Path(source_receipt['original29_reference_path']))
    actual_root = after['body_pos_w'][:, 0]
    root_delta = actual_root - original['source_qpos29'][:, :3]
    assert np.max(np.linalg.norm(root_delta, axis=1)) < .20
    hand_ids = np.array([native.body(name).id - 1 for name in ('left_wrist_roll_rubber_hand', 'right_wrist_roll_rubber_hand', 'torso_link')])
    quats = before['body_quat_w'][:, hand_ids]
    rotations = Rotation.from_quat(quats[:, :, [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(len(lift), 3, 3, 3)
    offsets = np.array(((.264, -.025, 0), (.264, .025, 0), (0, 0, .35)))
    before_tasks = before['body_pos_w'][:, hand_ids] + np.einsum('ntij,tj->nti', rotations, offsets)
    after_tasks = before_tasks.copy(); after_tasks[:, :, 2] += lift[:, None]
    before_relative = before_tasks - before['body_pos_w'][:, :1]
    after_relative = after_tasks - actual_root[:, None]
    relative_change = float(np.max(np.abs(after_relative - before_relative)))
    assert relative_change < 1e-12
    world_task_errors = np.linalg.norm(after_tasks - original['source_task_position_w'], axis=-1)
    relative_task_errors = np.linalg.norm(after_relative - (original['source_task_position_w'] - original['source_qpos29'][:, None, :3]), axis=-1)
    transform = dict(schema_version=1, kind='causal_upward_whole_pose_translation', clip=clip,
        input_reference_file='before_floor_reference.npz', input_reference_sha256=sha(output / 'before_floor_reference.npz'),
        input_report_sha256=sha(output / 'before_floor_report.json'), output_reference_sha256=sha(output / 'reference.npz'),
        frame_lift_file='frame_lift.npz', frame_lift_sha256=sha(output / 'frame_lift.npz'),
        filter=dict(kind='critically_damped_exact_discrete', omega_per_second=OMEGA, fixed_buffer_m=BUFFER,
            clearance_guard_epsilon_m=CLEARANCE_EPS, dt_seconds=DT, initial_filtered_lift='first_required_lift',
            initial_filter_velocity_mps=0., pose_preview_frames=0, guard='max(raw_required+epsilon, filtered_required+buffer)',
            guard_active_frames=int(guard.sum()), parameters_identical_across_all_four_clips=True,
            parameter_selection='Fixed tested candidate; buffer20mm selected for zero guard activations and added acceleration below2m/s2 on the provided four recordings.'),
        pose_prefix_causality_tests_passed=True, derivative_convention='Central difference of transformed body Z positions; unchanged existing XY derivatives.',
        exported_linear_velocity_future_pose_support_frames=1, exported_linear_velocity_future_pose_support_seconds=.02,
        native_model_sha256=sha(native_path), physics_contract_sha256=sha(ROOT / PHYSICS),
        floor_geom_id=floor, foot_geom_ids=spheres, foot_sphere_radii_m=native.geom_size[np.array(spheres), 0].tolist(),
        lift_min_m=float(lift.min()), lift_max_m=float(lift.max()), lift_p95_m=float(np.percentile(lift, 95)),
        added_vertical_velocity_max_mps=float(np.max(np.abs(np.diff(lift) / DT))),
        added_vertical_acceleration_max_mps2=float(np.max(np.abs(np.diff(lift, 2) / DT**2))),
        initial_foot_clearance_before_m=before_clearance[0].tolist(), initial_foot_clearance_after_m=after_clearance[0].tolist(),
        foot_clearance_after_min_m=float(after_clearance.min()), median_min_foot_clearance_after_m=float(np.median(after_clearance.min(axis=1))),
        already_clear_frames=int(np.sum(before_clearance.min(axis=1) > 0)),
        already_clear_frame_added_lift_max_m=float(lift[before_clearance.min(axis=1) > 0].max()),
        geometric_both_feet_clear_2mm_before=int(np.sum(before_clearance.min(axis=1) > .002)),
        geometric_both_feet_clear_2mm_after=int(np.sum(after_clearance.min(axis=1) > .002)),
        geometric_clear_frames_preserved=bool(np.all(after_clearance[before_clearance.min(axis=1) > .002] > .002)),
        near_floor_2mm_transitions_before=[int(np.count_nonzero(np.diff(before_clearance[:, side] <= .002))) for side in (0, 1)],
        near_floor_2mm_transitions_after=[int(np.count_nonzero(np.diff(after_clearance[:, side] <= .002))) for side in (0, 1)],
        source_contact_labels_present=False, geometric_clearance_is_not_ballistic_flight=True,
        root_relative_hand_head_geometry_change_max_m=relative_change,
        original29_world_root_error_p95_m=float(np.percentile(np.linalg.norm(root_delta, axis=1), 95)),
        original29_world_root_error_max_m=float(np.max(np.linalg.norm(root_delta, axis=1))),
        original29_root_z_error_max_m=float(np.max(np.abs(root_delta[:, 2]))),
        physical_floor_changed=False, root_relative_intent_preserved=True, joints_unchanged=True, source_timing_changed=False,
        self_collision_geometry_changed=False, foot_floor_clear_only=True, dynamics_qualified=False, hardware_authorized=False)
    (output / 'floor_transform_receipt.json').write_text(json.dumps(transform, indent=2, allow_nan=False))
    floor_metadata = {key: transform[key] for key in ('kind','lift_min_m','lift_max_m','physical_floor_changed','root_relative_intent_preserved','joints_unchanged','source_timing_changed')}
    floor_metadata.update(transform_receipt_file='floor_transform_receipt.json', transform_receipt_sha256=sha(output / 'floor_transform_receipt.json'),
                          frame_lift_file='frame_lift.npz', frame_lift_sha256=sha(output / 'frame_lift.npz'))
    report = dict(kind='explicit_common_z_floor_reference_v4', clip=clip, frames=len(lift), full_original_timeline=True,
        reference_sha256=sha(output / 'reference.npz'), source_timing_scale=1., original_source_reference_unchanged=True,
        before_floor_reference_sha256=sha(output / 'before_floor_reference.npz'), before_floor_report_sha256=sha(output / 'before_floor_report.json'),
        reference_floor_transform=floor_metadata, original_world_hand_head_p95_m=np.percentile(world_task_errors, 95, axis=0).tolist(),
        original_root_relative_hand_head_p95_m=np.percentile(relative_task_errors, 95, axis=0).tolist(),
        original29_world_root_error_max_m=transform['original29_world_root_error_max_m'],
        foot_floor_clear=True, guard_active_frames=int(guard.sum()), retarget_geometry_changed=True,
        dynamics_qualified=False, full_body_tracking_qualified=False, hardware_authorized=False)
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    load_case_motion(clip, output / 'reference.npz')
    for path in output.iterdir():
        if path.is_file():
            copy_exact(path, portable / path.name)
    load_case_motion(clip, portable / 'reference.npz')
    receipt = dict(source_receipt)
    receipt.update(reference_path=str(portable / 'reference.npz'), reference_wsl_path=wsl(portable / 'reference.npz'),
        reference_sha256=sha(portable / 'reference.npz'), source_artifact_report_sha256=sha(output / 'report.json'),
        source_artifact_directory=str(output), source_artifact_reference_sha256=sha(output / 'reference.npz'),
        reference_floor_transform=floor_metadata, before_floor_reference_sha256=sha(portable / 'before_floor_reference.npz'),
        validation_authority=dict(function='load_case_motion', module_path=str(AUTHORITY), module_sha256=hashlib.sha256(AUTHORITY_BYTES).hexdigest(),
                                  snapshot_file='../validation_authority_snapshot.py', mujoco=mujoco.__version__),
        copied_files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in portable.iterdir() if p.is_file()},
        foot_floor_clear_only=True, dynamics_qualified=False, full_body_tracking_qualified=False,
        source_timing_changed=False, original_source_reference_unchanged=True)
    receipt['validation'] = dict(source_receipt['validation'], foot_collision_clearance_nonnegative=True,
        original_root_world_error_below_20cm=True, common_z_transform_numeric_invariance=True)
    (portable / 'portable_receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False))
    entry = dict(clip=clip, frames=len(lift), source_controls=source_receipt['source_requested_controls'],
        reference_wsl_path=wsl(portable / 'reference.npz'), reference_sha256=sha(portable / 'reference.npz'),
        portable_receipt_wsl_path=wsl(portable / 'portable_receipt.json'), portable_receipt_sha256=sha(portable / 'portable_receipt.json'),
        lift_max_m=transform['lift_max_m'], added_acceleration_max_mps2=transform['added_vertical_acceleration_max_mps2'],
        guard_active_frames=int(guard.sum()), all_frames_validated=True)
    entries.append(entry)
    print(json.dumps(entry), flush=True)
assert AUTHORITY.read_bytes() == AUTHORITY_BYTES
for directory in (RAW, PORTABLE):
    (directory / 'manifest.json').write_text(json.dumps(dict(kind='explicit_floor_clearance_reference_v4_pack', clips=entries,
        floor_transform_script_sha256=sha(__file__), original_sources_unchanged=True, source_timing_changed=False,
        physical_floor_changed=False, full_body_tracking_qualified=False, dynamics_qualified=False, hardware_authorized=False), indent=2))
