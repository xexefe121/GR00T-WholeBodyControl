"""Read the existing C++ stage-one controller as an offline training contract.

This module grants no deployment authority. In particular, an effort table
copied from C++ remains an unverified hardware assumption, not a motor rating.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re

from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES, HARDWARE_JOINT_IDS

HEADER = Path("gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/true23_active_gantry_core.hpp")
SIM_CONFIG = Path("gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json")


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


@dataclass(frozen=True)
class NativeSupportActuationProfile(StageOneActuationProfile):
    """An explicit simulator hypothesis, never a replacement hardware profile.

    Native configured gains, full SONIC targets, 5 rad/s slew and 35 Nm ankle
    model cap. Every feasible target stays at 95% of the existing quarter-
    effort guard. These choices require separate physical review and training.
    """

    @classmethod
    def from_sim_config(cls, path: Path) -> NativeSupportActuationProfile:
        raw = path.read_bytes()
        value = json.loads(raw)
        if (
            value.get("kind") != "g1_true23_mujoco_sim2sim_config"
            or value.get("robot_model") != "g1_23dof_rev_1_0"
        ):
            raise ValueError("native support requires the exact true23 simulator config")
        physics = value["physics"]
        if physics["control_decimation"] != 10 or physics["timestep_s"] != 0.002:
            raise ValueError("native support requires 500 Hz physics and 50 Hz policy")
        effort = list(physics["effort_limit_hardware_nm"])
        if len(effort) != 23:
            raise ValueError("native support effort vector must have 23 joints")
        for index in (4, 5, 10, 11):
            effort[index] = min(effort[index], 35.0)
        return cls(
            source_sha256=hashlib.sha256(raw).hexdigest(),
            kp=tuple(physics["kp_hardware"]),
            kd=tuple(physics["kd_hardware"]),
            effort=tuple(effort),
            joint_scale=(1.0,) * 23,
            fraction=1.0,
            slew_rad_s=5.0,
            timestep_s=0.002,
            target_margin_rad=0.05,
            hold_kp_fraction=0.25,
        )

    def contract(self) -> dict:
        return {
            **super().contract(),
            "kind": "g1_true23_native_support_projected_training_v1",
            "source_kind": "native_simulation_config_not_physical_controller",
            "effort_status": "unverified_simulator_table_with_35Nm_ankle_cap_not_manufacturer_rating",
            "effort_target_projection": True,
            "projection_guard_fraction": 0.95 * 0.25,
            "empty_intersection_response": "latch_training_termination_and_zero_effort_until_next_50Hz_reset",
            "gain_review_for_hardware_complete": False,
        }
