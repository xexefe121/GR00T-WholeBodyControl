"""Read the existing C++ stage-one controller as an offline training contract.

This module grants no deployment authority. In particular, an effort table
copied from C++ remains an unverified hardware assumption, not a motor rating.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import re

from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES, HARDWARE_JOINT_IDS


HEADER = Path("gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/true23_active_gantry_core.hpp")


def _scalar(source: str, name: str) -> float:
    match = re.search(r"\b" + re.escape(name) + r"\s*=\s*([0-9.eE+-]+)\s*;", source)
    if match is None:
        raise ValueError(f"missing numeric C++ constant: {name}")
    return float(match.group(1))


def _array(source: str, name: str) -> tuple[float, ...]:
    match = re.search(r"\b" + re.escape(name) + r"\s*=\s*\{([^}]+)\}", source)
    if match is None:
        raise ValueError(f"missing C++ array: {name}")
    body = re.sub(r"//[^\n]*", "", match.group(1))
    return tuple(float(item.strip()) for item in body.split(",") if item.strip())


def read_joint_amplitude_scale(source: str) -> tuple[float, ...]:
    # The committed pre-taper controller has no per-joint multiplier. Read
    # that historical behavior as identity; a referenced/malformed new array
    # must still fail instead of silently becoming an identity transform.
    uncommented = re.sub(r"//[^\n]*|/\*.*?\*/", "", source, flags=re.DOTALL)
    if "kStageOneJointAmplitudeScale" not in uncommented:
        return (1.0,) * 23
    return _array(uncommented, "kStageOneJointAmplitudeScale")


@dataclass(frozen=True)
class StageOneActuationProfile:
    source_sha256: str
    kp: tuple[float, ...]
    kd: tuple[float, ...]
    effort: tuple[float, ...]
    joint_scale: tuple[float, ...]
    fraction: float
    slew_rad_s: float
    timestep_s: float
    target_margin_rad: float
    hold_kp_fraction: float

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", self.source_sha256):
            raise ValueError("actuation profile requires source SHA256")
        for name in ("kp", "kd", "effort", "joint_scale"):
            values = getattr(self, name)
            if len(values) != 23 or any(not math.isfinite(value) or value <= 0 for value in values):
                raise ValueError(f"invalid 23-joint actuation {name}")
        if any(value > 1 for value in self.joint_scale):
            raise ValueError("joint scale cannot exceed one")
        for name in ("fraction", "slew_rad_s", "timestep_s", "target_margin_rad", "hold_kp_fraction"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid actuation {name}")
        if self.fraction > 1 or self.hold_kp_fraction > 1 or self.timestep_s != 0.002:
            raise ValueError("unsupported stage-one control timing or fractions")

    @classmethod
    def from_cpp(cls, header: Path) -> StageOneActuationProfile:
        raw = header.read_bytes()
        source = raw.decode("utf-8")
        # This version models the explicit quarter-effort armed guard. Refuse
        # changed/unsupported C++ expressions rather than silently using 0.25.
        if not re.search(
            r"std::abs\(predicted_effort\)\s*>\s*0\.25\s*\*\s*kHardwareEffortLimitNm\[compact\]", source
        ):
            raise ValueError("unsupported C++ predicted-effort guard; update the simulator contract")
        return cls(
            source_sha256=hashlib.sha256(raw).hexdigest(),
            kp=_array(source, "kStageOneKp"),
            kd=_array(source, "kStageOneKd"),
            effort=_array(source, "kHardwareEffortLimitNm"),
            joint_scale=read_joint_amplitude_scale(source),
            fraction=_scalar(source, "kStageOneActionFraction"),
            slew_rad_s=_scalar(source, "kStageOneTargetRateRadPerSecond"),
            timestep_s=_scalar(source, "kControlPeriodSeconds"),
            target_margin_rad=_scalar(source, "kTargetLimitMarginRad"),
            hold_kp_fraction=_scalar(source, "kPreArmHoldKpFraction"),
        )

    def contract(self) -> dict:
        return {
            "kind": "g1_true23_stage_one_actuation_training_v1",
            **asdict(self),
            "hardware_joint_names": list(HARDWARE_23_JOINT_NAMES),
            "hardware_motor_slots": list(HARDWARE_JOINT_IDS),
            "control_decimation": 10,
            "predicted_effort_guard_fraction": 0.25,
            "raw_to_safe_target_applications": 1,
            "previous_action_semantics": "unscaled_safe_native23",
            "integrator": "euler",
            "guard_response": "terminate_training_episode_not_hardware_recovery",
            "effort_status": "unverified_cpp_table_not_manufacturer_rating",
            "hardware_authorized": False,
            "deployment_ready": False,
        }
