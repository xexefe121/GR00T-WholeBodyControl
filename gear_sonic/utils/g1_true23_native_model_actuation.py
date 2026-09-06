"""Explicit nominal *simulation* actuator contract; never a hardware profile.

Targets are held at 50 Hz. PD is recomputed and motor effort saturated at
500 Hz. This deliberately does not inherit gantry target projection or slew.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES


@dataclass(frozen=True)
class NativeModelActuationProfile:
    source_sha256: str
    kp: tuple[float, ...]
    kd: tuple[float, ...]
    effort: tuple[float, ...]
    velocity: tuple[float, ...]
    armature: tuple[float, ...]
    damping: tuple[float, ...]
    frictionloss: tuple[float, ...]
    timestep_s: float = 0.002
    decimation: int = 10

    def __post_init__(self):
        for name in ("kp", "kd", "effort", "velocity", "armature", "damping", "frictionloss"):
            values = getattr(self, name)
            if len(values) != 23 or any(isinstance(v, bool) or not math.isfinite(v) for v in values):
                raise ValueError(f"{name} requires 23 finite hardware-order values")
            if any(v < 0 or (name in {"kp", "effort", "velocity"} and v == 0) for v in values):
                raise ValueError(f"invalid {name}")
        if self.timestep_s != 0.002 or type(self.decimation) is not int or self.decimation != 10:
            raise ValueError("native-model requires 500 Hz physics and 50 Hz policy")
        if len(self.source_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.source_sha256):
            raise ValueError("profile requires source SHA256")

    @classmethod
    def from_sim_config(cls, path: Path):
        raw = path.read_bytes()
        payload = json.loads(raw)
        if (
            payload.get("kind") != "g1_true23_mujoco_sim2sim_config"
            or payload.get("robot_model") != "g1_23dof_rev_1_0"
        ):
            raise ValueError("not the native23 simulator configuration")
        physics = payload["physics"]
        if payload.get("control_hz") != 50 or physics.get("integrator") != "Euler":
            raise ValueError("native-model requires 50 Hz control and Euler physics")
        fields = {
            name: tuple(physics[key])
            for name, key in {
                "kp": "kp_hardware",
                "kd": "kd_hardware",
                "effort": "effort_limit_hardware_nm",
                "velocity": "velocity_limit_hardware_radps",
                "armature": "armature_hardware",
                "damping": "joint_damping_hardware",
                "frictionloss": "joint_frictionloss_hardware",
            }.items()
        }
        return cls(
            hashlib.sha256(raw).hexdigest(),
            **fields,
            timestep_s=physics["timestep_s"],
            decimation=physics["control_decimation"],
        )

    def contract(self):
        return {
            "kind": "g1_true23_nominal_native_model_actuation_v1",
            "source_sha256": self.source_sha256,
            "joint_names_hardware": list(HARDWARE_23_JOINT_NAMES),
            **{
                name: list(getattr(self, name))
                for name in ("kp", "kd", "effort", "velocity", "armature", "damping", "frictionloss")
            },
            "timestep_s": self.timestep_s,
            "control_decimation": self.decimation,
            "target_transform": "safe_target_transform_once_per_policy_step",
            "previous_action": "transformed_normalized_requested_native23_target",
            "pd": "kp*(held_target-measured_q)-kd*measured_dq",
            "effort_saturation": "symmetric_full_configured_simulation_effort_each_substep",
            "target_slew_projection": False,
            "quarter_effort_projection": False,
            "requested_effort_and_saturation_recorded": True,
            "limits_verified_on_hardware": False,
            "simulator_only": True,
            "hardware_authorized": False,
            "deployment_ready": False,
        }


def native_model_pd_numpy(target, q, dq, profile: NativeModelActuationProfile):
    """Return requested torque, saturated torque, invalid rows and excess cost."""
    target, q, dq = (np.asarray(x) for x in (target, q, dq))
    if target.shape != q.shape or q.shape != dq.shape or q.shape[-1:] != (23,):
        raise ValueError("PD inputs require matching [...,23] shapes")
    requested = np.asarray(profile.kp) * (target - q) - np.asarray(profile.kd) * dq
    invalid = ~np.isfinite(requested).all(axis=-1)
    effort = np.asarray(profile.effort)
    applied = np.where(np.expand_dims(invalid, -1), 0.0, np.clip(requested, -effort, effort))
    excess = np.maximum(np.abs(requested) / effort - 1.0, 0.0)
    cost = np.square(np.nan_to_num(excess, nan=10, posinf=10, neginf=10).clip(0, 10)).mean(axis=-1)
    return requested, applied, invalid, cost


def native_model_pd_torch(target, q, dq, profile: NativeModelActuationProfile):
    import torch

    if target.shape != q.shape or q.shape != dq.shape or q.shape[-1:] != (23,):
        raise ValueError("PD inputs require matching [...,23] shapes")
    requested = q.new_tensor(profile.kp) * (target - q) - q.new_tensor(profile.kd) * dq
    invalid = ~torch.isfinite(requested).all(dim=-1)
    effort = q.new_tensor(profile.effort)
    applied = torch.where(invalid.unsqueeze(-1), 0.0, requested.clamp(-effort, effort))
    excess = (requested.abs() / effort - 1).clamp_min(0)
    cost = torch.nan_to_num(excess, nan=10, posinf=10, neginf=10).clamp(0, 10).square().mean(dim=-1)
    return requested, applied, invalid, cost
