import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from gear_sonic.scripts.retime_g1_true23_sonic_reference import main, retime_reference
from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE
from gear_sonic.utils.g1_23dof_task_space_retarget import build_mjlab_motion_arrays
from gear_sonic.utils.g1_true23_sonic_library_replay import validate_library_motion

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT.parent / "GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml"


@pytest.fixture
def materials():
    if not MODEL_PATH.is_file():
        pytest.skip("native23 model unavailable")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    joints = np.tile(SAFE_TARGET_DEFAULT_Q_HARDWARE, (16, 1))
    joints[:, 16] += np.linspace(0, 0.2, 16)
    path = SimpleNamespace(
        root_pos_w=np.tile([0.0, 0.0, 0.8], (16, 1)),
        root_quat_wxyz=np.tile([1.0, 0.0, 0.0, 0.0], (16, 1)),
        joint_pos_hardware=joints,
        fps=50.0,
    )
    return model, build_mjlab_motion_arrays(model, path)


def test_identity_copies_every_original_channel(materials):
    model, source = materials
    arrays, phase, actual = retime_reference(source, model, 1)
    assert actual == 1
    np.testing.assert_array_equal(phase, np.arange(16))
    for key in source:
        np.testing.assert_array_equal(arrays[key], source[key])
        assert not np.shares_memory(arrays[key], source[key])


def test_half_speed_preserves_all_joint_samples_and_recomputes_fk_velocity(materials):
    model, source = materials
    before = {key: value.copy() for key, value in source.items()}
    arrays, phase, actual = retime_reference(source, model, 2)
    assert actual == 2 and validate_library_motion(arrays) == 31
    np.testing.assert_array_equal(arrays["joint_pos"][::2], source["joint_pos"])
    # Source FK was stored before pose quantization; rebuilt FK uses the
    # stored float32 root/joint samples. Allow two float32 absolute epsilons.
    np.testing.assert_allclose(
        arrays["body_pos_w"][::2], source["body_pos_w"], atol=2 * np.finfo(np.float32).eps, rtol=0
    )
    np.testing.assert_array_equal(phase[::2], np.arange(16))
    expected_velocity = (np.diff(arrays["joint_pos"].astype(np.float64), axis=0) * 50).astype(np.float32)
    np.testing.assert_array_equal(arrays["joint_vel"][1:], expected_velocity)
    assert np.mean(arrays["joint_vel"][1:, 16]) == pytest.approx(
        np.mean(source["joint_vel"][1:, 16]) / 2, abs=1e-6
    )
    for key in source:
        np.testing.assert_array_equal(source[key], before[key])


@pytest.mark.parametrize("scale", [0.5, 4.1, float("nan"), float("inf")])
def test_invalid_scale_rejected(materials, scale):
    model, source = materials
    with pytest.raises(ValueError, match="time scale"):
        retime_reference(source, model, scale)


def test_cli_saves_standard_motion_and_exclusive_audit_sidecar(materials, monkeypatch, tmp_path):
    _, source = materials
    input_path, output = tmp_path / "source.npz", tmp_path / "slow.npz"
    np.savez_compressed(input_path, **source)
    monkeypatch.setattr(
        "sys.argv",
        [
            "retime",
            "--motion",
            str(input_path),
            "--model",
            str(MODEL_PATH),
            "--time-scale",
            "2",
            "--output",
            str(output),
        ],
    )
    assert main() == 0
    with np.load(output) as archive:
        assert validate_library_motion(dict(archive)) == 31
    report = json.loads(output.with_suffix(".json").read_text())
    assert report["original_samples_present_at_exact_phases"] == 16
    assert not report["original_tempo_parity"] and not report["dynamic_feasibility_proven"]
    assert not report["hardware_authorized"]
    with pytest.raises(FileExistsError):
        main()
