import copy
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from gear_sonic.scripts import simulate_g1_sonic_library_motions as legacy
from gear_sonic.utils.g1_sonic_cpp_parameters import CppParameters, capture_cpp_parameters, file_sha256


@pytest.fixture(scope="module")
def captured(tmp_path_factory):
    root = Path(__file__).resolve().parents[2]
    header = root / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp"
    return capture_cpp_parameters(header, tmp_path_factory.mktemp("sonic_cpp") / "path with spaces")


def test_captured_parameters_match_header_damping_and_preserve_sources(captured):
    parameters, report = captured
    assert report["comparison_to_legacy"]["kds"]["different_beyond_float_roundoff_indices"] == [
        4,
        5,
        10,
        11,
        13,
        14,
    ]
    np.testing.assert_allclose(parameters.kds[[4, 5, 10, 11, 13, 14]], 2 * legacy.DAMPING_5020, rtol=1e-7)
    for name in ("kps", "action_scale", "default_angles"):
        assert report["comparison_to_legacy"][name]["different_beyond_float_roundoff_indices"] == []
    assert all(file_sha256(path) == digest for path, digest in report["inputs"].items())
    assert not report["complete_cpp_deployment_equivalence_proven"]
    assert not report["hardware_authorized"]


def test_target_matches_compiled_cpp_including_permutation_and_final_float_cast(captured):
    parameters, report = captured
    rng = np.random.default_rng(37)
    for action in (np.zeros(29), np.linspace(-9.9, 9.9, 29), *rng.uniform(-9, 9, (8, 29))):
        source = " ".join(str(float(x)) for x in action.astype(np.float32))
        output = subprocess.run(
            [report["binary"], "--targets"], input=source, check=True, text=True, capture_output=True, timeout=10
        )
        expected = np.asarray(json.loads(output.stdout))
        np.testing.assert_array_equal(parameters.target(action), expected)
        np.testing.assert_array_equal(expected, expected.astype(np.float32).astype(float))


@pytest.mark.parametrize(
    "damage", ["shape", "nan", "negative", "precision", "permutation", "fractional_index", "effort"]
)
def test_corrupt_parameter_payload_rejected(captured, damage):
    _, report = captured
    payload = json.loads(Path(report["parameters_path"]).read_text())
    if damage == "shape":
        payload["kps"].pop()
    elif damage == "nan":
        payload["action_scale"][0] = float("nan")
    elif damage == "negative":
        payload["kds"][0] = -1
    elif damage == "precision":
        payload["kds"][0] += 1e-12
    elif damage == "permutation":
        payload["isaaclab_to_mujoco"] = list(range(29))
    elif damage == "fractional_index":
        payload["isaaclab_to_mujoco"][0] = 0.5
    else:
        payload["motor_effort_constants"][0] *= 2
    with pytest.raises(ValueError):
        CppParameters(payload)


def test_parameter_copies_cannot_alias_input_or_legacy(captured):
    _, report = captured
    payload = json.loads(Path(report["parameters_path"]).read_text())
    original = copy.deepcopy(payload)
    parameters = CppParameters(payload)
    payload["kds"][4] = 99
    assert parameters.kds[4] == original["kds"][4]
    with pytest.raises(ValueError):
        parameters.kds[4] = 99
    assert legacy.KDS[4] == legacy.DAMPING_5020


def test_existing_capture_is_not_overwritten(captured):
    _, report = captured
    binary = Path(report["binary"])
    with pytest.raises(FileExistsError):
        capture_cpp_parameters(binary.parent / "policy_parameters.hpp", binary.parent)
    assert all(file_sha256(path) == digest for path, digest in report["inputs"].items())
