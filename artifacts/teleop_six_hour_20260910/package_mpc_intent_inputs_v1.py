"""Copy and fully validate the frozen v3 multistart references for MPC use.

Only artifacts are written. Controllers, source motion, exported references,
and original export receipts are never modified.
"""
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
ARCHIVE = Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910')
OUTPUT = ARCHIVE / 'mjbatch_intent_inputs_v1'
SOURCE = BASE / 'intent_retarget_v3_multistart'
AUTHORITY = ROOT / 'gear_sonic/scripts/evaluate_g1_true23_bfmzero.py'
MODEL_HELPER = ROOT / 'gear_sonic/utils/g1_true23_step1b_mujoco.py'
AUTHORITY_BYTES = AUTHORITY.read_bytes()
MODEL_HELPER_BYTES = MODEL_HELPER.read_bytes()
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
from gear_sonic.scripts.evaluate_g1_true23_bfmzero import DATA, MODEL, PHYSICS, load_motion, load_case_motion


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wsl(path):
    value = str(Path(path).absolute()).replace('\\', '/')
    assert value[1:3] == ':/'
    return '/mnt/' + value[0].lower() + '/' + value[3:]


def copy_exact(source, destination):
    if destination.exists():
        assert sha(source) == sha(destination), str(destination)
    else:
        shutil.copy2(source, destination)
    assert sha(source) == sha(destination)
    return dict(sha256=sha(destination), bytes=destination.stat().st_size)


assert OUTPUT.resolve().is_relative_to(ARCHIVE.resolve())
OUTPUT.mkdir(exist_ok=True)
assert AUTHORITY.read_bytes() == AUTHORITY_BYTES
assert MODEL_HELPER.read_bytes() == MODEL_HELPER_BYTES
(OUTPUT / 'validation_authority_snapshot.py').write_bytes(AUTHORITY_BYTES)
(OUTPUT / 'model_validation_helper_snapshot.py').write_bytes(MODEL_HELPER_BYTES)
copy_exact(Path(__file__), OUTPUT / 'packaging_script_snapshot.py')

native_model_path = ROOT.parent / 'GR00T-WholeBodyControl' / MODEL
source_model_path = ROOT.parent / 'GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_29dof.xml'
physics_path = ROOT / PHYSICS
native = mujoco.MjModel.from_xml_path(str(native_model_path))
limits = native.jnt_range[1:].copy()
velocity = np.asarray(json.loads(physics_path.read_text())['physics']['velocity_limit_hardware_radps'])
copied_names = ('reference.npz', 'report.json', 'original_timeline.json', 'solver_failures.json',
                'retarget_runner_snapshot.py', 'arm_ik_snapshot.py', 'hand_geometry_snapshot.py',
                'physics_contract_snapshot.json')
snapshot_for_module = {
    'g1_true23_intent_retarget.py': 'retarget_runner_snapshot.py',
    'g1_true23_intent_arm_ik.py': 'arm_ik_snapshot.py',
    'g1_true23_hand_frame_tasks.py': 'hand_geometry_snapshot.py',
}
entries = []
for clip in ('pico', 'walk003', 'walk008', 'walk002'):
    source = SOURCE / clip
    destination = OUTPUT / clip
    destination.mkdir(exist_ok=True)
    source_report = json.loads((source / 'report.json').read_text())
    assert source_report['kind'].endswith('_v3_multistart')
    assert source_report['clip'] == clip and source_report['full_original_timeline'] is True
    assert source_report['source_timing_scale'] == 1.0
    files = {name: copy_exact(source / name, destination / name) for name in copied_names}
    assert files['reference.npz']['sha256'] == source_report['reference_sha256']

    # This authority checks every exported body pose, joint position, linear
    # derivative and world angular interval against native topology and timing.
    motion, timeline, reference_path = load_case_motion(clip, destination / 'reference.npz')
    original, original_timeline, original_path = load_motion(clip)
    copied_timeline = json.loads((destination / 'original_timeline.json').read_text())
    assert timeline == original_timeline == copied_timeline
    original29_path = DATA / ('pico_freedancing_v1/optical_reference_v2/original29.npz' if clip == 'pico' else
                              f'{clip}/original_source_bundle_v1/original_reference.npz')
    provenance_checks = {}
    for path, digest in source_report['source_provenance'].items():
        item = Path(path)
        if item.name in snapshot_for_module:
            item = destination / snapshot_for_module[item.name]
        provenance_checks[str(item)] = sha(item) == digest
    assert all(provenance_checks.values())
    assert source_report['source_provenance'][str(original_path)] == sha(original_path)
    assert source_report['source_provenance'][str(original29_path)] == sha(original29_path)
    native_bundle_reference = BASE / 'mjbatch_native23_inputs_v1' / clip / 'native_original.npz'
    assert sha(native_bundle_reference) == sha(original_path)

    frames = len(motion['joint_pos'])
    phase = next(row for row in timeline['phases'] if row['name'] == 'source_motion')
    assert frames == source_report['frames'] == timeline['total_frames']
    assert frames - timeline['prehistory_frames'] == timeline['total_requested_controls']
    assert phase['requested_controls'] == timeline['source_frames']
    assert phase['control_stop'] - phase['control_start'] == phase['requested_controls']
    assert phase['frame_stop'] - phase['frame_start'] == phase['requested_controls']
    with np.load(original29_path, allow_pickle=False) as archive:
        assert len(archive['source_qpos29']) == frames
        original29_has_fps = 'fps' in archive.files
        if original29_has_fps:
            assert np.array_equal(archive['fps'], motion['fps'])
    assert float(motion['fps'][0]) == 50.
    bounds_max = float(max(0., np.max(np.maximum(limits[:, 0] - motion['joint_pos'], motion['joint_pos'] - limits[:, 1]))))
    adjacent_ratio = float(np.max(np.abs(np.diff(motion['joint_pos'], axis=0)) * 50 / velocity))
    quat_error = float(np.max(np.abs(np.linalg.norm(motion['body_quat_w'], axis=-1) - 1.)))
    assert bounds_max <= 1e-8 and adjacent_ratio <= 1 + 1e-8 and quat_error <= 1e-5
    flags = {name: True for name in (
        'all_frames_checked', 'finite_fields', 'same_fields_shapes_as_original', 'fps_50',
        'full_original_timeline', 'source_phase_counts_unchanged', 'unit_body_quaternions',
        'native_joint_bounds', 'adjacent_joint_speed_limits', 'joint_vel_matches_timed_positions',
        'body_lin_vel_matches_timed_positions', 'body_ang_vel_matches_world_rotation_intervals',
        'native_body_fk_positions', 'native_body_fk_rotations', 'copied_bytes_match_source',
        'source_provenance_matches')}
    receipt = dict(
        schema_version=1, kind='native23_intent_retarget_portable_receipt_v1', clip=clip,
        reference_file='reference.npz', reference_path=str(reference_path), reference_wsl_path=wsl(reference_path),
        reference_sha256=sha(reference_path), original_native_reference_sha256=sha(original_path),
        original29_reference_sha256=sha(original29_path), source_artifact_report_sha256=sha(source / 'report.json'),
        source_artifact_directory=str(source), source_artifact_reference_sha256=sha(source / 'reference.npz'),
        original_native_reference_path=str(original_path), original_native_reference_wsl_path=wsl(original_path),
        original29_reference_path=str(original29_path), original29_reference_wsl_path=wsl(original29_path),
        base_native_bundle_reference_sha256=sha(native_bundle_reference),
        base_native_bundle_reference_wsl_path=wsl(native_bundle_reference),
        native_model_sha256=sha(native_model_path), original29_model_sha256=sha(source_model_path),
        physics_contract_sha256=sha(physics_path), fps=50, frame_count=frames,
        original29_archive_has_fps=original29_has_fps,
        source_clock_authority='unchanged original native reference fps and lifecycle timeline',
        prehistory_frames=timeline['prehistory_frames'], total_requested_controls=timeline['total_requested_controls'],
        source_requested_controls=phase['requested_controls'], source_phase=phase,
        phase_counts=[{key: row[key] for key in ('name', 'control_start', 'control_stop', 'frame_start', 'frame_stop', 'requested_controls')}
                      for row in timeline['phases']],
        timeline_file='original_timeline.json', timeline_sha256=sha(destination / 'original_timeline.json'),
        fields={key: dict(shape=list(value.shape), dtype=str(value.dtype)) for key, value in motion.items()},
        validation=flags, validation_authority=dict(function='load_case_motion', module_path=str(AUTHORITY),
            module_sha256=hashlib.sha256(AUTHORITY_BYTES).hexdigest(), snapshot_file='../validation_authority_snapshot.py',
            model_helper_sha256=hashlib.sha256(MODEL_HELPER_BYTES).hexdigest(), mujoco=mujoco.__version__),
        validation_numbers=dict(native_joint_range_excess_max_rad=bounds_max,
            adjacent_joint_velocity_ratio_max=adjacent_ratio, body_quaternion_norm_error_max=quat_error),
        source_provenance_checks=provenance_checks, copied_files=files,
        physical_initialization_motion='retarget_reference_frame_10',
        recorded_target_seed_validation_motion='original_native_reference',
        seed_initial_state_equality_to_retarget_required=False, seed_physical_states_may_be_copied=False,
        reference_geometry_changed=True, source_timing_changed=False, source_timing_scale=1.,
        reference_is_kinematic_only=True, full_body_tracking_qualified=False, dynamics_qualified=False,
        hardware_authorized=False)
    (destination / 'portable_receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False))
    entries.append(dict(clip=clip, reference_wsl_path=wsl(reference_path), reference_sha256=sha(reference_path),
                        portable_receipt_wsl_path=wsl(destination / 'portable_receipt.json'),
                        portable_receipt_sha256=sha(destination / 'portable_receipt.json'),
                        frames=frames, source_controls=phase['requested_controls'], validation_passed=True))
    print(json.dumps(entries[-1]), flush=True)

assert AUTHORITY.read_bytes() == AUTHORITY_BYTES
assert MODEL_HELPER.read_bytes() == MODEL_HELPER_BYTES
(OUTPUT / 'manifest.json').write_text(json.dumps(dict(kind='native23_intent_mpc_portable_pack_v1', clips=entries,
    packaging_script_sha256=sha(__file__), packaging_script_snapshot_file='packaging_script_snapshot.py',
    validator_snapshot_sha256=sha(OUTPUT / 'validation_authority_snapshot.py'),
    controller_code_modified=False, hardware_authorized=False), indent=2))
print(json.dumps(dict(manifest=str(OUTPUT / 'manifest.json'), clips=len(entries))), flush=True)
