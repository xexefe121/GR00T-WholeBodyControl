from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from gear_sonic.scripts import record_g1_sonic_public29_baseline as module
from gear_sonic.scripts.prepare_g1_true23_twist2_replay import native23_poses
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256


def test_import_retains_every_original29_joint_and_matches_existing_times():
    recording = dict(
        fps=30.0,
        root_pos=np.arange(36).reshape(12, 3) * 0.001,
        root_rot=np.tile([0.0, 0.0, 0.0, 1.0], (12, 1)),
        dof_pos=np.arange(348).reshape(12, 29) * 0.001,
    )
    selected, times, queries = native23_poses(recording)
    poses = module.source29_poses(recording)
    np.testing.assert_array_equal(poses[:, :7], selected[:, :7])
    np.testing.assert_array_equal(poses[:, 7:][:, module.SOURCE_MJ29_KEEP_INDICES], selected[:, 7:])
    for j in range(29):
        np.testing.assert_array_equal(poses[:, 7 + j], np.interp(queries, times, recording["dof_pos"][:, j]))


def test_full_source_and_return_timing_is_not_shortened():
    standing = np.r_[[0, 0, 0.76, 1, 0, 0, 0], module.stock.DEFAULT_ANGLES]
    source = np.tile(standing, (31, 1))
    source[:, 0] = np.arange(31) * 0.001
    source[:, 7:] += np.arange(29) * 0.0001
    before = source.copy()
    poses, phases = module.lifecycle29(source, standing)
    phase = next(p for p in phases if p["name"] == "source_motion")
    assert len(poses) == 31 + 750 + 11
    assert phase == dict(name="source_motion", control_start=350, control_stop=381)
    np.testing.assert_array_equal(poses[361:392], source)
    np.testing.assert_array_equal(source, before)
    np.testing.assert_array_equal(poses[-1, :2], source[-1, :2])
    np.testing.assert_array_equal(poses[-10:], np.repeat(poses[-1:], 10, axis=0))


@pytest.mark.parametrize("invalid", ("shape", "nonfinite", "root_quaternion", "standing_quaternion"))
def test_lifecycle_rejects_invalid_original_source(invalid):
    standing = np.r_[[0, 0, 0.76, 1, 0, 0, 0], module.stock.DEFAULT_ANGLES]
    source = np.tile(standing, (12, 1))
    if invalid == "shape":
        source = source[:, :30]
    elif invalid == "nonfinite":
        source[0, 7] = np.nan
    elif invalid == "root_quaternion":
        source[0, 3] = 0.5
    else:
        standing[3] = 0.5
    with pytest.raises(ValueError):
        module.lifecycle29(source, standing)


@pytest.fixture
def material():
    model = mujoco.MjModel.from_xml_path(
        str(Path(__file__).resolve().parents[2] / "gear_sonic_deploy/g1/scene_29dof.xml")
    )
    model.opt.timestep = 0.002
    poses = np.tile(np.r_[[0, 0, 0.78, 1, 0, 0, 0], module.stock.DEFAULT_ANGLES], (23, 1))
    parameters = SimpleNamespace(
        default_angles=module.stock.DEFAULT_ANGLES.astype(float),
        kps=module.stock.KPS,
        kds=module.stock.KDS,
        effort=module.stock.EFFORT_LIMITS,
        target=lambda action: module.stock.DEFAULT_ANGLES.astype(float),
    )
    return model, poses, parameters


def test_actual_simulation_and_received_semantics_preserve_all29_state(material):
    model, poses, parameters = material
    teacher = SimpleNamespace(infer=lambda semantic, history: np.zeros(29, dtype=np.float32))
    before = compiled_model_sha256(model)
    arrays, received, metrics = module.run_case(
        model, model, parameters, teacher, poses, [dict(name="source_motion", control_start=0, control_stop=12)]
    )
    assert metrics["completed_controls"] == metrics["requested_controls"] == 12
    assert arrays["qpos"].shape == (13, 36)
    assert arrays["physics_post_qpos"].shape == (120, 36)
    np.testing.assert_array_equal(arrays["physics_pre_qpos"][1:], arrays["physics_post_qpos"][:-1])
    np.testing.assert_array_equal(arrays["qpos"][1:], arrays["physics_post_qpos"][9::10])
    np.testing.assert_allclose(
        arrays["source_timestamps_s"][:, 0] - arrays["source_timestamps_s"][:, 1], 0.2, rtol=0, atol=1e-15
    )
    for t, encoder in enumerate(arrays["encoder267"]):
        q = received["joint_pos"][9 + t : 20 + t, :12]
        np.testing.assert_array_equal(encoder[:120], q[:-1].ravel())
        np.testing.assert_array_equal(encoder[120:240], ((q[1:] - q[:-1]) / np.float32(0.02)).ravel())
    assert (
        not metrics["hardware_authorized"]
        and not metrics["deployment_ready"]
        and not metrics["native23_qualified"]
    )
    assert compiled_model_sha256(model) == before


def test_invalid_first_action_retains_zero_control_failure(material):
    model, poses, parameters = material
    teacher = SimpleNamespace(infer=lambda *args: np.full(29, 10.0, dtype=np.float32))
    arrays, _, metrics = module.run_case(
        model, model, parameters, teacher, poses, [dict(name="source_motion", control_start=0, control_stop=12)]
    )
    assert metrics["completed_controls"] == 0
    assert metrics["requested_controls"] == 12
    assert metrics["failure"]["type"] == "RuntimeError"
    assert not metrics["source_landmark_screen_passed"]
    assert arrays["qpos"].shape == (1, 36)


def test_nonconstant_tail_cannot_be_silently_padded(material):
    model, poses, parameters = material
    poses[-1, 0] += 0.01
    with pytest.raises(ValueError, match="constant standing tail"):
        module.run_case(model, model, parameters, None, poses, [])
