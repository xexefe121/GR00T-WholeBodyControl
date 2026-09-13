"""Stationary control diagnostic, not a replacement dance/teleop reference.

Compare paired SONIC to the pinned standing compatibility actor in the same
native23 plant and unchanged outer envelope. Retain a standing-only observation
dataset if the actor completes. No training or physical authorization here.
"""

import json
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort

from gear_sonic.scripts import evaluate_g1_true23_deployment_envelope as envelope
from gear_sonic.scripts.record_g1_sonic_original29_baseline import dump
from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE
from gear_sonic.utils.g1_sonic_cpp_observation_trace import audit_engine_trace
from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256
from gear_sonic.utils.g1_true23_actuation_profile import NativeSupportActuationProfile
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import (
    CleanTrue23MujocoController,
    UnitreeZeroVelocityFallbackPolicy,
    encoder267_from_reference,
    motion_reference_terms,
)
from gear_sonic.utils.g1_true23_interior_target_filter import run_interior_case
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256
from gear_sonic.utils.g1_true23_sim_acquisition import simulate_balance_transition
from gear_sonic.utils.g1_true23_step1b_mujoco import term_major_history

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ASSETS = ROOT.parent / "GR00T-WholeBodyControl"
OUTPUT = HERE / "stationary"
source = json.loads((HERE / "report.json").read_text())
inputs = {**source["inputs"], str(HERE / "report.json"): file_sha256(HERE / "report.json")}
flags = dict(teacher_accepted=False, hardware_authorized=False, deployment_ready=False)


def bind(path):
    path = Path(path).resolve(strict=True)
    digest = file_sha256(path)
    if inputs.get(str(path), digest) != digest:
        raise ValueError(f"stationary source changed: {path}")
    inputs[str(path)] = digest
    return path


profile = NativeSupportActuationProfile.from_sim_config(bind(ROOT / envelope.PHYSICS))
pair = source["pair"]
policy = envelope._policy(
    Path(pair["encoder"]["path"]),
    Path(pair["decoder"]["path"]),
    pair["decoder"]["sha256"],
    encoder_hash=pair["encoder"]["sha256"],
)
options = ort.SessionOptions()
options.intra_op_num_threads = options.inter_op_num_threads = 1
balance_path = bind(
    ASSETS / "artifacts/external/unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx"
)
balance = UnitreeZeroVelocityFallbackPolicy(balance_path, session_options=options)
controller = CleanTrue23MujocoController(
    model_path=ASSETS / envelope.MODEL, physics_path=ROOT / envelope.PHYSICS, policy=None
)
q, quat = np.array(SAFE_TARGET_DEFAULT_Q_HARDWARE), np.array([1.0, 0.0, 0.0, 0.0])
height, placement = envelope.measured_ground_contact_height(controller, q, quat)
controller.reset(base_position=np.array([0.0, 0.0, height]), base_quaternion_wxyz=quat, joint_position_hardware=q)
controller.module.mj_forward(controller.model, controller.data)
motion = dict(
    fps=np.array([50.0]),
    joint_pos=np.tile(q, (511, 1)),
    joint_vel=np.zeros((511, 23)),
    body_pos_w=np.tile(controller.data.xpos[1:], (511, 1, 1)),
    body_quat_w=np.tile(controller.data.xquat[1:], (511, 1, 1)),
    body_lin_vel_w=np.zeros((511, 24, 3)),
    body_ang_vel_w=np.zeros((511, 24, 3)),
)
OUTPUT.mkdir(exist_ok=False)
with (OUTPUT / "stationary_reference.npz").open("xb") as stream:
    np.savez_compressed(stream, **motion)
bind(OUTPUT / "stationary_reference.npz")
bind(Path(__file__))
for name, module in list(sys.modules.items()):
    path = getattr(module, "__file__", None)
    if name.startswith("gear_sonic.") and path and Path(path).suffix == ".py":
        bind(path)
dump(OUTPUT / "started.json", {"inputs": dict(inputs), "stationary_only": True, **flags})
records = []
for acquisition in (False, True):
    label = "sonic_after_acquisition" if acquisition else "sonic_synthetic_start"
    result, arrays = run_interior_case(
        fraction=1.0,
        root=ROOT,
        asset_root=ASSETS,
        policy=policy,
        motion=motion,
        kp=np.asarray(profile.kp),
        kd=np.asarray(profile.kd),
        joint_scale=np.ones(23),
        ankle_effort=35.0,
        slew_rate=5.0,
        initial_state="reference",
        maximum_steps=None,
        startup_hold_s=5.0 if acquisition else 0.0,
        return_hold_s=5.0 if acquisition else 0.0,
        transition_policy=balance if acquisition else None,
        align_reference_start=acquisition,
        project_transition_effort=acquisition,
        project_active_effort=True,
        stateful_native_controller=True,
        trace_active_actuation=True,
    )
    with (OUTPUT / f"{label}.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    bind(OUTPUT / f"{label}.npz")
    records.append({"name": label, "result": result, **flags})
    dump(OUTPUT / f"{label}.json", records[-1])
    bind(OUTPUT / f"{label}.json")
    print(
        json.dumps(
            {
                "case": label,
                "completed": result["completed_transitions"],
                "requested": result["requested_transitions"],
                "failure": result["failure"],
                "return": result["return_hold"].get("completed_transitions"),
            }
        ),
        flush=True,
    )

physics = {
    key: []
    for key in (
        "pre_qpos",
        "post_qpos",
        "pre_qvel",
        "post_qvel",
        "requested_effort",
        "actual_effort",
        "engine_pre_time_s",
        "engine_post_time_s",
        "engine_warning_counts",
    )
}
original_module = controller.module


class PhysicsRecorder:
    def __getattr__(self, name):
        return getattr(original_module, name)

    def mj_step(self, model, data):
        assert model is controller.model and data is controller.data
        for key, value in (("pre_qpos", data.qpos), ("pre_qvel", data.qvel), ("requested_effort", data.ctrl)):
            physics[key].append(value.copy())
        physics["engine_pre_time_s"].append(float(data.time))
        original_module.mj_step(model, data)
        physics["engine_post_time_s"].append(float(data.time))
        physics["engine_warning_counts"].append([int(item.number) for item in data.warning])
        for key, value in (
            ("post_qpos", data.qpos),
            ("post_qvel", data.qvel),
            ("actual_effort", data.qfrc_actuator[6:]),
        ):
            physics[key].append(value.copy())


observations = {key: [] for key in ("encoder267", "history930", "target_hardware23", "kp", "kd")}


class CapturingStandingActor:
    def reset(self):
        balance.reset()

    def activate(self, q):
        balance.activate(q)

    def infer(self, **kwargs):
        packet = motion_reference_terms(motion, 9)
        encoder = encoder267_from_reference(packet, controller.buffered_robot_pelvis_q9)
        history = term_major_history(controller.history)
        target, kp, kd = balance.infer(**kwargs)
        for key, value in (
            ("encoder267", encoder),
            ("history930", history),
            ("target_hardware23", target),
            ("kp", kp),
            ("kd", kd),
        ):
            observations[key].append(value.copy())
        return target, kp, kd


controller.module = PhysicsRecorder()
controller.history = [controller._policy_frame().copy() for _ in range(10)]
result, states, last_target = simulate_balance_transition(
    controller, CapturingStandingActor(), duration_s=10.0, slew_rate=5.0, previous_target=q, project_effort=True
)
arrays = {"physics_" + key: np.asarray(value) for key, value in physics.items()}
arrays.update({key: np.asarray(value) for key, value in observations.items()})
arrays.update(physics_dt=np.array([0.002]), qpos=states, final_target=last_target)
engine = audit_engine_trace(arrays)
result.update(
    actual_engine_audit=engine,
    compiled_native_model_sha256=compiled_model_sha256(controller.model),
    stationary_teacher_candidate_only=True,
    sonic_policy_used=False,
    all_dance_or_teleop_readiness_proven=False,
    **flags,
)
with (OUTPUT / "standing_compatibility_actor.npz").open("xb") as stream:
    np.savez_compressed(stream, **arrays)
bind(OUTPUT / "standing_compatibility_actor.npz")
records.append({"name": "standing_compatibility_actor", "result": result, **flags})
print(
    json.dumps(
        {
            "case": "standing_compatibility_actor",
            "completed": result["completed_transitions"],
            "standing": result["existing_guard_screen_passed"],
            "engine": engine["passed"],
        }
    ),
    flush=True,
)
for path in list(inputs):
    bind(path)
dump(
    OUTPUT / "report.json",
    {
        "kind": "g1_true23_stationary_actor_comparison_diagnostic_v1",
        "inputs": inputs,
        "records": records,
        "placement": placement,
        "stationary_only": True,
        "full_motion_suite_not_replaced": True,
        "pair": pair,
        **flags,
    },
)
