"""Read-only C++ parameter capture for offline SONIC29 comparisons.

This validates the arithmetic/command boundary, not C++ observations, timing,
DDS, firmware torque clipping, mode transitions, or native23 hardware safety.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from gear_sonic.scripts import simulate_g1_sonic_library_motions as legacy


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class CppParameters:
    def __init__(self, payload):
        if payload.get("kind") != "g1_sonic_cpp_policy_parameters_v1":
            raise ValueError("expected a captured SONIC C++ parameter dump")
        for name in ("kps", "kds", "action_scale", "default_angles"):
            value = np.array(payload[name], dtype=np.float64, copy=True)
            if value.shape != (29,) or not np.isfinite(value).all():
                raise ValueError("C++ parameters require finite hardware29 arrays")
            if name != "default_angles" and np.any(value <= 0):
                raise ValueError("C++ gains and action scales must be positive")
            if name in ("kps", "kds") and not np.array_equal(value, value.astype(np.float32).astype(float)):
                raise ValueError("C++ gains must preserve their float32 command representation")
            value.setflags(write=False)
            setattr(self, name, value)
        for name, expected in (
            ("isaaclab_to_mujoco", legacy.MUJOCO_TO_ISAAC_INDEX),
            ("mujoco_to_isaaclab", legacy.ISAAC_TO_MUJOCO_INDEX),
        ):
            actual = np.asarray(payload[name])
            if actual.dtype.kind not in "iu" or not np.array_equal(actual, expected):
                raise ValueError("unsupported C++/legacy joint permutation")
        constants = np.asarray(payload["motor_effort_constants"], dtype=float)
        if not np.array_equal(constants, [25, 88, 139, 5]):
            raise ValueError("legacy simulation torque limits differ from the captured motor constants")
        # The legacy loop's torque clipping remains a simulation assumption.
        self.effort = legacy.EFFORT_LIMITS.copy()
        self.effort.setflags(write=False)

    def target(self, action):
        action = np.asarray(action)
        if action.shape != (29,) or not np.isfinite(action).all() or np.max(np.abs(action)) >= 10:
            raise ValueError("C++ target arithmetic requires a finite bounded 29-element action")
        float_action = action.astype(np.float32).astype(np.float64)
        value = self.default_angles + float_action[legacy.MUJOCO_TO_ISAAC_INDEX] * self.action_scale
        return value.astype(np.float32).astype(np.float64)

    def comparison(self):
        pairs = {
            "kps": legacy.KPS,
            "kds": legacy.KDS,
            "action_scale": legacy.ACTION_SCALE,
            "default_angles": legacy.DEFAULT_ANGLES.astype(float),
        }
        return {
            name: {
                "legacy": old.tolist(),
                "cpp": getattr(self, name).tolist(),
                "maximum_absolute_difference": float(np.max(np.abs(getattr(self, name) - old))),
                "different_beyond_float_roundoff_indices": np.flatnonzero(
                    ~np.isclose(getattr(self, name), old, atol=1e-7, rtol=1e-7)
                ).tolist(),
            }
            for name, old in pairs.items()
        }


def capture_cpp_parameters(header, output, *, compiler="c++"):
    """Compile the checked-in arithmetic witness against an exact header copy.

    `output` must be new. Only this diagnostic directory is written. A compiler
    executable is invoked without a shell; no deployment target is built.
    """
    header, output = Path(header).resolve(strict=True), Path(output).resolve()
    if header.name != "policy_parameters.hpp":
        raise ValueError("capture requires the explicit SONIC policy header")
    source = Path(__file__).resolve().parents[1] / "scripts/cpp/dump_sonic_policy_parameters.cpp"
    compiler_path = shutil.which(compiler)
    if compiler_path is None:
        raise ValueError("a local C++ compiler is required for parameter capture")
    compiler_path = Path(compiler_path).resolve(strict=True)
    identities = {str(path): file_sha256(path) for path in (header, source, compiler_path)}
    output.mkdir(parents=True, exist_ok=False)
    snapshot_header, snapshot_source = output / header.name, output / source.name
    for origin, target in ((header, snapshot_header), (source, snapshot_source)):
        with target.open("xb") as stream:
            stream.write(origin.read_bytes())
        if file_sha256(target) != identities[str(origin)]:
            raise ValueError("parameter source changed while capturing it")
    binary = output / "dump_sonic_policy_parameters"
    command = [
        str(compiler_path),
        "-std=c++17",
        "-O2",
        "-ffp-contract=off",
        str(snapshot_source),
        "-o",
        str(binary),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=10)
    payload = json.loads(result.stdout)
    parameters = CppParameters(payload)
    parameter_path = output / "parameters.json"
    with parameter_path.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    for path, digest in identities.items():
        if file_sha256(path) != digest:
            raise ValueError("parameter capture input changed during compilation")
    for path in (snapshot_header, snapshot_source, binary, parameter_path):
        identities[str(path)] = file_sha256(path)
    return parameters, {
        "inputs": identities,
        "compile_command": command,
        "compiler_version": subprocess.run(
            [str(compiler_path), "--version"], check=True, capture_output=True, text=True, timeout=10
        ).stdout,
        "binary": str(binary),
        "parameters_path": str(parameter_path),
        "comparison_to_legacy": parameters.comparison(),
        "complete_cpp_deployment_equivalence_proven": False,
        "teacher_accepted": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
