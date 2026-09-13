"""Read-only independent FK, velocity, timeline and provenance audit of exports."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gear_sonic.scripts.evaluate_g1_true23_bfmzero import ROOT, DATA, MODEL, PHYSICS, load_motion, load_case_motion
from gear_sonic.utils.g1_true23_hand_frame_tasks import neutral_wrist_hand_tasks

BASE = ROOT / 'artifacts/teleop_six_hour_20260910'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(clip):
    directory = BASE / 'intent_retarget_v2' / clip
    report = json.loads((directory / 'report.json').read_text())
    old, timeline, _ = load_motion(clip)
    motion, override_timeline, _ = load_case_motion(clip, directory / 'reference.npz')
    with np.load(directory / 'kinematic_diagnostics.npz', allow_pickle=False) as z:
        recorded = {key: z[key].copy() for key in z.files}
    original_path = DATA / ('pico_freedancing_v1/optical_reference_v2/original29.npz' if clip == 'pico' else
                            f'{clip}/original_source_bundle_v1/original_reference.npz')
    with np.load(original_path, allow_pickle=False) as z:
        original = {key: z[key].copy() for key in z.files}
    native = mujoco.MjModel.from_xml_path(str(ROOT.parent / 'GR00T-WholeBodyControl' / MODEL))
    source = mujoco.MjModel.from_xml_path(str(ROOT.parent / 'GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_29dof.xml'))
    tasks, convention = neutral_wrist_hand_tasks(source, native)
    tasks = [next(t for t in tasks if t.name == name) for name in ('left_hand', 'right_hand', 'head_proxy')]
    nd, sd = mujoco.MjData(native), mujoco.MjData(source)
    nf = [native.body(side + '_ankle_roll_link').id for side in ('left', 'right')]
    sf = [source.body(side + '_ankle_roll_link').id for side in ('left', 'right')]
    nt, st = native.body('torso_link').id, source.body('torso_link').id
    qpos = np.column_stack((motion['body_pos_w'][:, 0], motion['body_quat_w'][:, 0], motion['joint_pos']))
    count = len(qpos)
    scalar = dict(body_position_fk_max_m=0., body_rotation_fk_max_rad=0., torso_rotation_original_max_rad=0.,
                  torso_xy_original_max_m=0., torso_height_original_max_m=0., recorded_qpos_max_abs=0.,
                  foot_position_diagnostic_reconstruction_max_m=0., foot_rotation_diagnostic_reconstruction_max_rad=0.,
                  task_error_diagnostic_reconstruction_max_m=0., original_source_task_fk_max_m=0.)
    torso_shift = []
    actual_task = []
    original_task = []
    for i in range(count):
        nd.qpos[:] = qpos[i]
        sd.qpos[:] = original['source_qpos29'][i]
        mujoco.mj_kinematics(native, nd)
        mujoco.mj_kinematics(source, sd)
        poserr = np.linalg.norm(nd.xpos[1:] - motion['body_pos_w'][i], axis=1)
        rot = Rotation.from_quat(nd.xquat[1:, [1, 2, 3, 0]])
        wanted = Rotation.from_quat(motion['body_quat_w'][i, :, [1, 2, 3, 0]].T)
        scalar['body_position_fk_max_m'] = max(scalar['body_position_fk_max_m'], float(poserr.max()))
        scalar['body_rotation_fk_max_rad'] = max(scalar['body_rotation_fk_max_rad'], float((rot.inv() * wanted).magnitude().max()))
        torso_delta = nd.xpos[nt] - sd.xpos[st]
        torso_shift.append(torso_delta)
        scalar['torso_rotation_original_max_rad'] = max(scalar['torso_rotation_original_max_rad'], float(Rotation.from_matrix(nd.xmat[nt].reshape(3, 3) @ sd.xmat[st].reshape(3, 3).T).magnitude()))
        feet = np.linalg.norm(nd.xpos[nf] - sd.xpos[sf], axis=1)
        angles = np.array([Rotation.from_matrix(nd.xmat[a].reshape(3, 3) @ sd.xmat[b].reshape(3, 3).T).magnitude() for a, b in zip(nf, sf)])
        scalar['foot_position_diagnostic_reconstruction_max_m'] = max(scalar['foot_position_diagnostic_reconstruction_max_m'], float(np.max(np.abs(feet - recorded['foot_position_error'][i]))))
        scalar['foot_rotation_diagnostic_reconstruction_max_rad'] = max(scalar['foot_rotation_diagnostic_reconstruction_max_rad'], float(np.max(np.abs(angles - recorded['foot_orientation_error'][i]))))
        actual = np.array([nd.xpos[native.body(t.target_body).id] + nd.xmat[native.body(t.target_body).id].reshape(3, 3) @ t.target_point for t in tasks])
        source_tasks = np.array([sd.xpos[source.body(t.source_body).id] + sd.xmat[source.body(t.source_body).id].reshape(3, 3) @ t.source_point for t in tasks])
        actual_task.append(actual)
        original_task.append(source_tasks)
        scalar['original_source_task_fk_max_m'] = max(scalar['original_source_task_fk_max_m'], float(np.max(np.linalg.norm(source_tasks - original['source_task_position_w'][i], axis=1))))
        errors = np.linalg.norm(actual - original['source_task_position_w'][i], axis=1)
        scalar['task_error_diagnostic_reconstruction_max_m'] = max(scalar['task_error_diagnostic_reconstruction_max_m'], float(np.max(np.abs(errors - recorded['original_task_error'][i]))))
    torso_shift = np.array(torso_shift)
    scalar['torso_xy_original_max_m'] = float(np.max(np.linalg.norm(torso_shift[:, :2], axis=1)))
    scalar['torso_height_original_max_m'] = float(np.max(np.abs(torso_shift[:, 2])))
    scalar['recorded_qpos_max_abs'] = float(np.max(np.abs(qpos - recorded['qpos'])))
    vlim = np.asarray(json.loads((ROOT / PHYSICS).read_text())['physics']['velocity_limit_hardware_radps'])
    forward = np.diff(motion['joint_pos'], axis=0) / .02
    worst = np.unravel_index(np.argmax(np.abs(forward) / vlim), forward.shape)
    scalar['export_joint_velocity_gradient_max_abs'] = float(np.max(np.abs(np.gradient(motion['joint_pos'], .02, axis=0) - motion['joint_vel'])))
    scalar['export_body_linear_velocity_gradient_max_abs'] = float(np.max(np.abs(np.gradient(motion['body_pos_w'], .02, axis=0) - motion['body_lin_vel_w'])))
    reconstruction = 0.
    earlier, later = np.maximum(np.arange(count) - 1, 0), np.minimum(np.arange(count) + 1, count - 1)
    duration = (later - earlier) * .02
    for body in range(24):
        rotations = Rotation.from_quat(motion['body_quat_w'][:, body][:, [1, 2, 3, 0]])
        reconstructed = Rotation.from_rotvec(motion['body_ang_vel_w'][:, body] * duration[:, None]) * rotations[earlier]
        reconstruction = max(reconstruction, float((reconstructed.inv() * rotations[later]).magnitude().max()))
    scalar['world_angular_interval_reconstruction_max_rad'] = reconstruction
    phase = next(p for p in timeline['phases'] if p['name'] == 'source_motion')
    sl = slice(phase['frame_start'], phase['frame_stop'])
    task_error = np.linalg.norm(np.array(actual_task) - np.array(original_task), axis=-1)
    result = dict(clip=clip, frames=count, source_frames=phase['requested_controls'], all_scalar_checks=scalar,
                  timeline_dict_matches_original=timeline == override_timeline == json.loads((directory / 'original_timeline.json').read_text()),
                  original_npz_fields={key: list(value.shape) for key, value in original.items()},
                  export_finite=all(np.isfinite(value).all() for value in motion.values()),
                  native_range_excess_max_rad=float(np.maximum(np.maximum(native.jnt_range[1:, 0] - motion['joint_pos'], motion['joint_pos'] - native.jnt_range[1:, 1]), 0).max()),
                  exported_velocity_ratio_max=float(np.max(np.abs(motion['joint_vel']) / vlim)),
                  forward_interval_velocity_ratio_max=float(np.max(np.abs(forward) / vlim)),
                  forward_interval_velocity_abs_max_rad_s=np.max(np.abs(forward), axis=0).tolist(),
                  worst_forward_interval=dict(frame0=int(worst[0]), frame1=int(worst[0]+1), hardware_joint=int(worst[1]), joint_name=native.joint(1 + int(worst[1])).name, speed_rad_s=float(forward[worst]), limit_rad_s=float(vlim[worst[1]])),
                  arm_forward_step_max_rad=float(np.max(np.abs(np.diff(motion['joint_pos'][:, 13:], axis=0)))),
                  torso_height_shift_range_m=[float(torso_shift[:, 2].min()), float(torso_shift[:, 2].max())],
                  original_task_source_phase_p95_m=np.percentile(task_error[sl], 95, axis=0).tolist(),
                  report_source_hashes_current={path: sha(path) == value for path, value in report['source_provenance'].items()},
                  output_hash_matches=sha(directory / 'reference.npz') == report['reference_sha256'],
                  snapshot_hash_matches=dict(retarget=sha(directory / 'retarget_runner_snapshot.py') == report['source_provenance'][str(ROOT / 'gear_sonic/utils/g1_true23_intent_retarget.py')],
                                             arm=sha(directory / 'arm_ik_snapshot.py') == report['source_provenance'][str(ROOT / 'gear_sonic/utils/g1_true23_intent_arm_ik.py')]),
                  geometry_helper_hash_omitted_from_original_receipt=sha(ROOT / 'gear_sonic/utils/g1_true23_hand_frame_tasks.py'),
                  hand_task_convention=convention)
    return result


if __name__ == '__main__':
    rows = [audit(clip) for clip in ('walk002', 'walk003', 'walk008', 'pico')]
    output = dict(kind='independent_read_only_retarget_v2_audit', source_file_sha256=sha(__file__), checks=rows,
                  dynamics_qualified=False, hardware_authorized=False)
    (BASE / 'intent_retarget_v2_independent_audit.json').write_text(json.dumps(output, indent=2, allow_nan=False))
    print(json.dumps([{key: row[key] for key in ('clip', 'frames', 'all_scalar_checks', 'native_range_excess_max_rad', 'exported_velocity_ratio_max', 'forward_interval_velocity_ratio_max', 'worst_forward_interval', 'original_task_source_phase_p95_m')} for row in rows], indent=2))
