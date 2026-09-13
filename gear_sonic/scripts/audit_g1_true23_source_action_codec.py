"""Read-only C++ source-action oracle versus native23 simulation codec."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from gear_sonic.utils.g1_23dof_contract import (
    HARDWARE_23_ACTION_SCALE,
    ISAACLAB_TO_MUJOCO_DOF,
    LOW_LATENCY_RELEASE_SHA256,
    MUJOCO_TO_ISAACLAB_DOF,
    NATIVE_IL23_TO_CANONICAL_IL29,
    SOURCE_MJ29_KEEP_INDICES,
)
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_DEFAULT_Q_HARDWARE,
    safe_target_transform_numpy,
)
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import sha256_file
from gear_sonic.utils.g1_true23_source_action_codec import (
    SOURCE_SCALE_NATIVE_IL23,
    source_action_codec_contract,
    source_action_history_numpy,
    source_scaled_precompensation,
)


def compile_source_oracle(asset_root):
    root = Path(__file__).resolve().parents[2]
    include = Path(asset_root) / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include"
    harness = root / "gear_sonic/tests/fixtures/sonic_source_action_oracle.cpp"
    with tempfile.TemporaryDirectory(prefix="sonic-source-action-") as temp:
        executable = Path(temp) / "oracle"
        subprocess.run(
            ["g++", "-std=c++17", "-O0", "-I", str(include), str(harness), "-o", str(executable)], check=True
        )
        return json.loads(subprocess.check_output([str(executable)], text=True))


def audit_source_oracle(oracle):
    keep = np.asarray(SOURCE_MJ29_KEEP_INDICES)
    select = np.asarray(NATIVE_IL23_TO_CANONICAL_IL29)
    mapping = np.asarray(oracle["source_il29_for_hardware29"])
    scales = np.asarray(oracle["scale_hardware29"])
    default = np.asarray(oracle["default_hardware29"])
    # Compare independent original29 C++ permutation against native23 row surgery.
    np.testing.assert_array_equal(mapping[keep], select[np.asarray(ISAACLAB_TO_MUJOCO_DOF)])
    np.testing.assert_array_equal(default[keep], SAFE_TARGET_DEFAULT_Q_HARDWARE)
    np.testing.assert_allclose(
        scales[keep][np.asarray(MUJOCO_TO_ISAACLAB_DOF)], SOURCE_SCALE_NATIVE_IL23, atol=1e-14, rtol=0
    )
    target_error = history_error = 0.0
    probes = np.vstack((np.zeros((1, 23)), np.eye(23) * 0.1, np.eye(23) * -0.1)).astype(np.float32)
    for raw in probes:
        source29 = np.zeros(29)
        source29[select] = raw
        expected = (default + source29[mapping] * scales)[keep]
        inverse, projection = source_scaled_precompensation(raw)
        safe, target = safe_target_transform_numpy(inverse)
        target_error = max(target_error, float(np.max(np.abs(expected - target))))
        np.testing.assert_allclose(projection, 0, atol=1e-14, rtol=0)
        history = np.zeros(930, np.float32)
        for frame in range(10):
            history[610 + frame * 29 + select] = safe
        decoded = source_action_history_numpy(history)
        expected_history = np.tile(source29, (10, 1))
        history_error = max(
            history_error, float(np.max(np.abs(decoded[610:900].reshape(10, 29) - expected_history)))
        )
    if target_error > 1e-7 or history_error > 1e-7:
        raise ValueError("source codec differs from independent original C++ equations")
    return dict(
        original_cpp_probe_count=len(probes),
        source_target_max_abs_error_rad=target_error,
        source_unprojected_history_max_abs_error=history_error,
        source_scale_hardware23=scales[keep].tolist(),
        legacy_native_scale_hardware23=list(HARDWARE_23_ACTION_SCALE),
        legacy_hip_pitch_displacement_ratio=float(HARDWARE_23_ACTION_SCALE[0] / scales[0]),
        retained_row_mapping_exact=True,
        retained_default_angles_exact=True,
        pass_source_action_equations=True,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    assets = args.asset_root.resolve(strict=True)
    root = Path(__file__).resolve().parents[2]
    inputs = {
        str(path): sha256_file(path)
        for path in (
            assets / "low_latency/last.pt",
            assets / "low_latency/config.yaml",
            assets / "gear_sonic/envs/manager_env/robots/g1.py",
            assets / "gear_sonic/envs/manager_env/modular_tracking_env_cfg.py",
            assets / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp",
            assets / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp",
            root / "gear_sonic/tests/fixtures/sonic_source_action_oracle.cpp",
            root / "gear_sonic/utils/g1_true23_source_action_codec.py",
            Path(__file__),
        )
    }
    if inputs[str(assets / "low_latency/last.pt")] != LOW_LATENCY_RELEASE_SHA256:
        raise ValueError("source checkpoint identity mismatch")
    import yaml

    config = yaml.safe_load((assets / "low_latency/config.yaml").read_text())
    if config["manager_env"]["config"]["robot"]["type"] != "g1_model_12_dex":
        raise ValueError("paired low-latency configuration does not select source motor model")
    oracle = compile_source_oracle(assets)
    report = dict(
        kind="native23_original_cpp_source_action_codec_audit_v1",
        source_robot_type="g1_model_12_dex",
        training_source_formula="G1_MODEL_12_ACTION_SCALE = 0.25 * effort_limit_sim / stiffness",
        training_source_and_release_config_hashed_not_embedded_in_weights=True,
        original_cpp=oracle,
        codec=source_action_codec_contract(),
        audit=audit_source_oracle(oracle),
        inputs=inputs,
        physics_steps=0,
        hardware_or_network_actuation_used=False,
        deployment_ready=False,
        simulator_qualified=False,
    )
    if any(sha256_file(Path(path)) != digest for path, digest in inputs.items()):
        raise ValueError("source action audit input changed during execution")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report["audit"], allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
