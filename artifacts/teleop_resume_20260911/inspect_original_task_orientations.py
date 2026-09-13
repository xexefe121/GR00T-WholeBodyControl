"""Original29 hand/head world and orientation errors on the qualified native trace."""

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gear_sonic.utils.g1_true23_hand_frame_tasks import neutral_wrist_hand_tasks

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    bundle = ROOT / 'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'
    trace_path = BASE / 'student_actual_oracle_control1_resume1001_v1/nominal/trace.npz'
    physics_path = BASE / 'expert_resumed_full_independent_physics_v1/report.json'
    intent_path = BASE / 'expert_resumed_full_independent_intent_v1/report.json'
    original_path = bundle / 'walk003/original29.npz'
    original_model_path = ROOT.parent / 'GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_29dof.xml'
    physics, intent = [json.loads(p.read_text()) for p in (physics_path, intent_path)]
    assert physics['independent_segment_pass'] and intent['full_lifecycle_source_intent_pass']
    assert sha(trace_path) in physics['input_hashes'].values() and sha(trace_path) in intent['hashes'].values()
    manifest = json.loads((bundle / 'manifest.json').read_text())
    assert sha(original_path) == manifest['cases']['walk003']['original29.npz']
    assert sha(bundle / 'native_prepared.xml') == manifest['portable_xml_sha256']
    assert sha(bundle / 'prepared_model_arrays.npz') == manifest['prepared_arrays_sha256']
    native = mujoco.MjModel.from_xml_path(str(bundle / 'native_prepared.xml'))
    with np.load(bundle / 'prepared_model_arrays.npz') as arrays:
        for key in arrays.files:
            getattr(native, key)[:] = arrays[key]
        mujoco.mj_setConst(native, mujoco.MjData(native))
    source_model = mujoco.MjModel.from_xml_path(str(original_model_path))
    tasks, convention = neutral_wrist_hand_tasks(source_model, native)
    names = ('left_hand', 'right_hand', 'head_proxy')
    tasks = [next(t for t in tasks if t.name == name) for name in names]
    correction = {v['name']: np.asarray(v['neutral_source_hand_rotation_in_roll']) for v in convention['hands']}
    with np.load(trace_path) as trace:
        control = trace['global_control']
        selected = (control >= 350) & (control < 1169)
        frames = trace['source_frame'][selected]
        poses = trace['qpos'][1:][selected]
    assert len(poses) == 819
    with np.load(original_path) as original:
        target = original['source_task_position_w'][frames]
        quats = original['source_task_quaternion_wxyz'][frames]
    actual, rotations = [], []
    data = mujoco.MjData(native)
    for pose in poses:
        data.qpos[:] = pose
        mujoco.mj_kinematics(native, data)
        position, orientation = [], []
        for task in tasks:
            body = native.body(task.target_body).id
            rotation = data.xmat[body].reshape(3, 3)
            position.append(data.xpos[body] + rotation @ task.target_point)
            orientation.append(rotation @ correction.get(task.name, np.eye(3)))
        actual.append(position)
        rotations.append(orientation)
    actual, rotations = np.asarray(actual), np.asarray(rotations)
    reference_rotations = Rotation.from_quat(quats.reshape(-1, 4)[:, [1, 2, 3, 0]])
    measured_rotations = Rotation.from_matrix(rotations.reshape(-1, 3, 3))
    orientation_error = np.degrees((reference_rotations.inv() * measured_rotations).magnitude()).reshape(819, 3)
    position_error = np.linalg.norm(actual - target, axis=-1)
    producer_path = BASE / 'student_actual_oracle_control1_resume1001_v1/outcome.json'
    producer = json.loads(producer_path.read_text())['nominal']['source_metrics']
    world_p95 = np.percentile(position_error, 95, axis=0)
    np.testing.assert_allclose(world_p95, producer['original_hand_head_world_p95_m'], atol=2e-14, rtol=0)
    out = BASE / 'expert_resumed_original_task_orientations_v1'
    out.mkdir(exist_ok=False)
    np.savez_compressed(out / 'task_errors.npz', source_frame=frames,
        original_position=target, actual_position=actual, original_quaternion_wxyz=quats,
        actual_rotation=rotations, world_position_error=position_error,
        orientation_error_deg=orientation_error)
    report = dict(kind='original29_task_world_and_orientation_inspection_no_dynamics',
        source_controls=819, physics_steps=0, inputs_bind_qualified_full_trace=True,
        position_world_p95_matches_producer=True,
        tasks={name: dict(world_position_p95_m=float(world_p95[i]),
            world_position_max_m=float(position_error[:, i].max()),
            orientation_geodesic_p50_deg=float(np.median(orientation_error[:, i])),
            orientation_geodesic_p95_deg=float(np.percentile(orientation_error[:, i], 95)),
            orientation_geodesic_max_deg=float(orientation_error[:, i].max())) for i, name in enumerate(names)},
        orientation_convention='Native hand roll frame composed with fixed source neutral-wrist rotation; original29 requested task orientation unchanged. Head uses torso orientation.',
        orientation_acceptance_threshold=None,
        missing_wrist_axes_not_reconstructed=True,
        original_source_position_gates_unchanged=True,
        hashes={str(p): sha(p) for p in (trace_path, physics_path, intent_path, original_path,
            original_model_path, producer_path, bundle / 'native_prepared.xml',
            bundle / 'prepared_model_arrays.npz', Path(__file__))})
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    (out / 'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report['tasks']))


if __name__ == '__main__':
    main()
