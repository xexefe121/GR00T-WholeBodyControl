"""Compile and call isolated, source-captured SONIC G1-mode observations.

Actual C++ gatherers, logger and math run on injected arrays. No G1Deploy,
SDK, DDS, CSV logging, command writer or physical mode transition is created.
This verifies a numerical boundary, NOT full deployment or planner parity.
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import yaml

from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256

METHODS = (
    "ComputeApplyDeltaHeading",
    "GatherMotionAnchorOrientationMutiFrame",
    "GatherMotionJointPositionsMultiFrame",
    "GatherMotionJointVelocitiesMultiFrame",
    "GatherHisBodyJointPositions",
    "GatherHisBodyJointVelocities",
    "GatherHisLastActions",
    "GatherHisBaseAngularVelocity",
    "GatherHisGravityDir",
    "GatherEncoderMode",
)
SOURCE_FILES = (
    "src/g1_deploy_onnx_ref.cpp",
    "src/state_logger.cpp",
    "src/file_sink.cpp",
    "include/state_logger.hpp",
    "include/file_sink.hpp",
    "include/math_utils.hpp",
    "include/policy_parameters.hpp",
    "include/robot_parameters.hpp",
    "include/utils.hpp",
    "include/motion_data_reader.hpp",
    "include/fk.hpp",
)
FLAGS = dict(teacher_accepted=False, hardware_authorized=False, deployment_ready=False)


def cpp_method(source, name):
    """Extract one complete method, ignoring braces inside comments/literals.

    This intentionally rejects raw C++ string literals rather than attempting
    an incomplete interpretation. Captured production methods have none.
    """
    match = list(re.finditer(r"^    (?:bool|std::array<double, 4>) " + re.escape(name) + r"\(", source, re.M))
    if len(match) != 1:
        raise ValueError(f"expected exactly one supported C++ method: {name}")
    start = match[0].start()
    # Parameter lists contain initializer-list defaults such as {}.
    cursor, parentheses = source.index("(", start), 0
    while cursor < len(source):
        char = source[cursor]
        parentheses += (char == "(") - (char == ")")
        cursor += 1
        if parentheses == 0:
            break
    opening = source.find("{", cursor)
    if opening < 0 or source[cursor:opening].strip():
        raise ValueError("C++ method has no body")
    cursor, depth, mode = opening, 0, "code"
    while cursor < len(source):
        char, pair = source[cursor], source[cursor : cursor + 2]
        if mode == "line":
            if char == "\n":
                mode = "code"
        elif mode == "block":
            if pair == "*/":
                mode = "code"
                cursor += 1
        elif mode in ('"', "'"):
            if char == "\\":
                cursor += 1
            elif char == mode:
                mode = "code"
        elif pair in ("//", "/*"):
            mode = "line" if pair == "//" else "block"
            cursor += 1
        elif pair == 'R"':
            raise ValueError("raw C++ string literal extraction is unsupported")
        elif char in ('"', "'"):
            mode = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start : cursor + 1]
        cursor += 1
    raise ValueError("unterminated C++ method")


def validate_observation_layout(source, config):
    registry = dict(re.findall(r'\{"([a-z0-9_]+)", (\d+),', source))
    encoder = config["encoder"]
    fields = [row["name"] for row in encoder["encoder_observations"] if row.get("enabled") is True]
    offset, offsets = 0, {}
    for name in fields:
        if name not in registry or name in offsets:
            raise ValueError("unknown or duplicate encoder field")
        offsets[name] = offset
        offset += int(registry[name])
    expected = {
        "encoder_mode_4": 0,
        "motion_joint_positions_10frame_step5": 4,
        "motion_joint_velocities_10frame_step5": 294,
        "motion_anchor_orientation_10frame_step5": 601,
    }
    modes = [mode for mode in encoder["encoder_modes"] if mode["mode_id"] == 0]
    names = [row["name"] for row in config["observations"] if row.get("enabled") is True]
    if (
        offset != 1762
        or encoder["dimension"] != 64
        or encoder["use_fp16"] is not False
        or len(modes) != 1
        or modes[0]["required_observations"] != list(expected)
        or any(offsets.get(name) != index for name, index in expected.items())
        or names
        != [
            "token_state",
            "his_base_angular_velocity_10frame_step1",
            "his_body_joint_positions_10frame_step1",
            "his_body_joint_velocities_10frame_step1",
            "his_last_actions_10frame_step1",
            "his_gravity_dir_10frame_step1",
        ]
        or [int(registry[name]) for name in names[1:]] != [30, 290, 290, 290, 30]
    ):
        raise ValueError("source configuration no longer matches the exact G1-mode 1762/994 witness ABI")
    return {"encoder_offsets": offsets, "encoder_size": offset, "decoder_size": 994, "history_size": 930}


def capture_cpp_observations(repository_root, output, *, compiler="c++"):
    root, output = Path(repository_root).resolve(strict=True), Path(output).resolve()
    source_root = root / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref"
    template = root / "gear_sonic/scripts/cpp/sonic_observation_witness.cpp"
    config_path = root / "gear_sonic_deploy/policy/release/observation_config.yaml"
    application = source_root / SOURCE_FILES[0]
    text = application.read_text()
    layout = validate_observation_layout(text, yaml.safe_load(config_path.read_text()))
    bodies = {name: cpp_method(text, name) for name in METHODS}
    compiler_path = shutil.which(compiler)
    if compiler_path is None:
        raise ValueError("a local C++20 compiler is required")
    compiler_path = Path(compiler_path).resolve(strict=True)
    origins = [
        *(source_root / name for name in SOURCE_FILES),
        template,
        config_path,
        compiler_path,
        Path(__file__),
    ]
    inputs = {str(path): file_sha256(path) for path in origins}
    output.mkdir(parents=True, exist_ok=False)
    copies = []
    for relative in SOURCE_FILES:
        original, copy = source_root / relative, output / relative
        copy.parent.mkdir(parents=True, exist_ok=True)
        with copy.open("xb") as stream:
            stream.write(original.read_bytes())
        if file_sha256(copy) != inputs[str(original)]:
            raise ValueError("source changed during capture")
        copies.append(copy)
    snapshot = output / template.name
    with snapshot.open("xb") as stream:
        stream.write(template.read_bytes())
    included = output / "captured_gatherers.inc"
    with included.open("x") as stream:
        stream.write("\n\n".join(bodies.values()) + "\n")
    snapshot_config = output / config_path.name
    with snapshot_config.open("xb") as stream:
        stream.write(config_path.read_bytes())
    binary = output / "sonic_observation_witness.so"
    command = [
        str(compiler_path),
        "-std=c++20",
        "-O2",
        "-ffp-contract=off",
        "-fPIC",
        "-shared",
        "-pthread",
        "-I",
        str(output / "include"),
        "-I",
        str(output),
        str(snapshot),
        str(output / "src/state_logger.cpp"),
        str(output / "src/file_sink.cpp"),
        "-o",
        str(binary),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    for path, digest in inputs.items():
        if file_sha256(path) != digest:
            raise ValueError("observation capture input changed")
    for path in [*copies, snapshot, included, snapshot_config, binary]:
        inputs[str(path)] = file_sha256(path)
    report = {
        "kind": "g1_sonic_cpp_observation_witness_v1",
        "inputs": inputs,
        "methods_captured_unchanged": list(bodies),
        "layout": layout,
        "binary": str(binary),
        "binary_sha256": file_sha256(binary),
        "compile_command": command,
        "compiler_version": subprocess.run(
            [str(compiler_path), "--version"], check=True, capture_output=True, text=True, timeout=10
        ).stdout,
        "csv_logging_enabled": False,
        "sdk_or_deployment_application_built": False,
        "sensor_values_and_reference_arrays_injected": True,
        "complete_cpp_deployment_or_planner_equivalence_proven": False,
        **FLAGS,
    }
    with (output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


def finite_array(value, shape):
    array = np.array(value, dtype=np.float64, order="C", copy=True)
    if (
        array.shape != shape
        or not np.isfinite(array).all()
        or np.max(np.abs(array), initial=0) > np.finfo(np.float32).max
    ):
        raise ValueError(f"expected a finite array of shape {shape}")
    return array


class CppObservations:
    """Single-threaded owned C++ witness; immutable references, injected state."""

    def __init__(self, capture, quaternion, joint, velocity, initial_robot_quaternion):
        self.pointer = None
        if (
            capture.get("kind") != "g1_sonic_cpp_observation_witness_v1"
            or any(capture.get(key) is not False for key in FLAGS)
            or file_sha256(capture["binary"]) != capture["binary_sha256"]
        ):
            raise ValueError("requires a hash-matched, explicitly unaccepted observation witness")
        self.frames = len(quaternion)
        if not 2 <= self.frames <= 100000:
            raise ValueError("observation witness needs 2..100000 reference frames")
        quaternion = finite_array(quaternion, (self.frames, 4))
        joint, velocity = finite_array(joint, (self.frames, 29)), finite_array(velocity, (self.frames, 29))
        initial = finite_array(initial_robot_quaternion, (4,))
        if not np.allclose(np.linalg.norm(quaternion, axis=1), 1, atol=1e-6, rtol=0) or not np.isclose(
            np.linalg.norm(initial), 1, atol=1e-6, rtol=0
        ):
            raise ValueError("reference and injected IMU quaternions must be unit rotations")
        library = ctypes.CDLL(capture["binary"])
        doubles = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        floats = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
        library.sonic_observation_create.argtypes = [ctypes.c_int, doubles, doubles, doubles, doubles]
        library.sonic_observation_create.restype = ctypes.c_void_p
        library.sonic_observation_destroy.argtypes = [ctypes.c_void_p]
        library.sonic_observation_destroy.restype = None
        library.sonic_observation_append.argtypes = [ctypes.c_void_p, doubles, doubles, doubles]
        library.sonic_observation_append.restype = ctypes.c_int
        library.sonic_observation_gather.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, floats, floats]
        library.sonic_observation_gather.restype = ctypes.c_int
        library.sonic_observation_error.restype = ctypes.c_char_p
        self.library = library
        self.pointer = library.sonic_observation_create(self.frames, quaternion, joint, velocity, initial)
        if not self.pointer:
            raise RuntimeError(library.sonic_observation_error().decode())

    def close(self):
        if self.pointer:
            self.library.sonic_observation_destroy(self.pointer)
            self.pointer = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def append(self, pose, velocity, action):
        if not self.pointer:
            raise ValueError("observation witness is closed")
        pose, velocity, action = (
            finite_array(pose, (36,)),
            finite_array(velocity, (35,)),
            finite_array(action, (29,)),
        )
        if not np.isclose(np.linalg.norm(pose[3:7]), 1, atol=1e-6, rtol=0):
            raise ValueError("injected state quaternion must be a unit rotation")
        if self.library.sonic_observation_append(self.pointer, pose, velocity, action):
            raise RuntimeError(self.library.sonic_observation_error().decode())

    def gather(self, frame, *, play=True):
        if not self.pointer or type(frame) is not int or not 0 <= frame < self.frames or type(play) is not bool:
            raise ValueError("observation witness needs an open handle, valid frame and explicit play boolean")
        encoder, history = np.empty(1762, dtype=np.float32), np.empty(930, dtype=np.float32)
        if self.library.sonic_observation_gather(self.pointer, frame, int(play), encoder, history):
            raise RuntimeError(self.library.sonic_observation_error().decode())
        if not np.isfinite(encoder).all() or not np.isfinite(history).all():
            raise ValueError("C++ observation is nonfinite")
        return encoder, history
