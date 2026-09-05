import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE
from gear_sonic.utils.g1_23dof_task_space_retarget import build_mjlab_motion_arrays
from gear_sonic.utils.g1_true23_reference_floor import (
    compiled_model_sha256,
    condition_reference_floor,
    motion_qpos,
    reference_geometry,
)


@pytest.fixture
def materials():
    path = Path(__file__).resolve().parents[2] / "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml"
    model = mujoco.MjModel.from_xml_path(str(path))
    joints = np.tile(SAFE_TARGET_DEFAULT_Q_HARDWARE, (32, 1))
    joints[:, 16] += np.linspace(0, 0.1, 32)
    trajectory = SimpleNamespace(
        root_pos_w=np.tile([0, 0, 0.74], (32, 1)),
        root_quat_wxyz=np.tile([1, 0, 0, 0], (32, 1)),
        joint_pos_hardware=joints,
        fps=50.0,
    )
    return model, build_mjlab_motion_arrays(model, trajectory)


def test_clearance_preserves_all_joint_samples_time_orientation_and_inputs(materials):
    model, source = materials
    before = {key: value.copy() for key, value in source.items()}
    model_hash = compiled_model_sha256(model)
    assert reference_geometry(model, source)["frames_with_floor_overlap"] > 0
    arrays, report = condition_reference_floor(source, {"mesh": model}, output_model=model)
    assert report["geometric_floor_clearance_passed"]
    assert report["after"]["mesh"]["frames_with_floor_overlap"] == 0
    assert report["source_frame_count"] == report["output_frame_count"] == 32
    assert report["frames_removed"] == 0 and report["time_scale"] == 1
    np.testing.assert_array_equal(arrays["joint_pos"], source["joint_pos"])
    for key in ("joint_vel", "body_quat_w", "body_ang_vel_w"):
        np.testing.assert_array_equal(arrays[key], source[key])
    np.testing.assert_array_equal(arrays["body_pos_w"][:, 0, :2], source["body_pos_w"][:, 0, :2])
    np.testing.assert_array_equal(arrays["body_quat_w"][:, 0], source["body_quat_w"][:, 0])
    for key in source:
        np.testing.assert_array_equal(source[key], before[key])
    assert compiled_model_sha256(model) == model_hash
    assert report["serialized_correction_audit"]["passed"]
    assert not report["dynamic_feasibility_proven"] and not report["support_contacts_or_com_optimized"]
    assert not report["deployment_ready"] and not report["hardware_authorized"]


def test_deep_penetration_cannot_be_hidden_by_dropping_frames(materials):
    model, motion = materials
    motion["body_pos_w"][:, :, 2] -= 0.5
    with pytest.raises(ValueError, match="lift bound|maximum lift"):
        condition_reference_floor(motion, {"mesh": model}, output_model=model)


def test_airborne_reference_is_not_lowered(materials):
    model, source = materials
    source["body_pos_w"][:, :, 2] += 1
    arrays, report = condition_reference_floor(source, {"mesh": model}, output_model=model)
    assert report["maximum_root_lift_m"] == 0
    np.testing.assert_array_equal(arrays["body_pos_w"][:, 0], source["body_pos_w"][:, 0])


def test_wrong_joint_layout_is_rejected(materials):
    _, motion = materials
    wrong = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>'
    )
    with pytest.raises(ValueError, match="native23"):
        motion_qpos(wrong, motion)


def test_inconsistent_body_channels_are_not_silently_rebuilt(materials):
    model, source = materials
    source["body_pos_w"][:, 5, 0] += 0.1
    with pytest.raises(ValueError, match="inconsistent"):
        condition_reference_floor(source, {"mesh": model}, output_model=model)


@pytest.mark.parametrize("value", [-1, 0, float("nan"), float("inf")])
def test_invalid_clearance_rejected(materials, value):
    model, motion = materials
    with pytest.raises(ValueError, match="bounds"):
        condition_reference_floor(motion, {"mesh": model}, output_model=model, clearance_m=value)


def test_compiled_model_hash_binds_collision_geometry(materials):
    model, _ = materials
    first = compiled_model_sha256(model)
    model.geom_size[0, 0] += 0.01
    assert compiled_model_sha256(model) != first


def test_cli_separates_input_asset_root_and_preserves_false_readiness(materials, tmp_path, monkeypatch):
    from gear_sonic.scripts import condition_g1_true23_reference_floor as cli

    model, source = materials
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    np.savez_compressed(asset_root / "source.npz", **source)
    manifest = tmp_path / "source_manifest.json"
    manifest.write_text(json.dumps({"motions": [{"name": "full_clip", "path": "source.npz", "weight": 2}]}))
    output = tmp_path / "conditioned"
    monkeypatch.setattr(cli, "build_training_geometry", lambda: (model, []))
    monkeypatch.setattr(
        "sys.argv",
        [
            "condition",
            "--manifest",
            str(manifest),
            "--asset-root",
            str(asset_root),
            "--output-dir",
            str(output),
            "--condition",
        ],
    )
    assert cli.main() == 0
    report = json.loads((output / "report.json").read_text())
    assert report["all_clips_geometrically_conditioned"] and report["clips_dropped"] == 0
    assert not report["deployment_ready"] and not report["dynamic_feasibility_proven"]
    assert report["records"][0]["source_path"] == str((asset_root / "source.npz").resolve())
    generated = json.loads((output / "motions.json").read_text())
    assert generated["motions"][0]["weight"] == 2
    with pytest.raises(FileExistsError):
        cli.main()
