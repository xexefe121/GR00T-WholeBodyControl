"""Static native23 actor observability audit; no dynamics integration or hardware.

Run from the transfer repository with its normal CPU dependency environment:
python artifacts/g1_true23_generalist/root_observability_20260907_v1/audit_root_observability.py

The three independent reset states are observation-unit-test fixtures, not a
rollout or evidence of motion performance. This script refuses to overwrite its
receipt and forbids every MuJoCo integration entry point on its controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import torch

from gear_sonic.envs.mjlab.sonic_true23 import (
    vr_3point_local_orn_target,
    vr_3point_local_target,
)
from gear_sonic.envs.mjlab.sonic_true23_causal_history import (
    causal_history_lower_body,
    causal_motion_anchor_ori_b,
)
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import (
    CleanTrue23MujocoController,
    encoder267_from_reference,
    motion_reference_terms,
)
from gear_sonic.utils.g1_true23_generalist_benchmark import (
    MODEL,
    PHYSICS,
    load_generalist_pair,
)
from gear_sonic.utils.g1_true23_step1b_mujoco import term_major_history


PINNED = {
    "motion": (
        "artifacts/g1_true23_frozen_lora/original29_neutral_hand_frame_fit_20260906_v1/happy_dance.native23.npz",
        "3fe0a95b3e33e291fffde0fc3ede0cfc2fbae3b5ca19e8a43fbbe72fe3d16251",
    ),
    "manifest": (
        "artifacts/g1_true23_generalist/smoke_20260907_v2/export/generalist.diagnostic.json",
        "ae19e00ed0441046f57ca3502390cf1c0106597b75f327e8e0a2035b84e92a5c",
    ),
    "checkpoint": (
        "artifacts/g1_true23_generalist/smoke_20260907_v2/checkpoints/native23_generalist_model_2.pt",
        "212b7ed4704dfd2f71f637a96a713407b622379e004838b0e423419aaf7904db",
    ),
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_identity(value: np.ndarray) -> dict:
    value = np.ascontiguousarray(value)
    return dict(
        shape=list(value.shape),
        dtype=str(value.dtype),
        c_contiguous_bytes_sha256=hashlib.sha256(value.tobytes()).hexdigest(),
    )


class NoIntegrationModule:
    def __init__(self, module):
        self.original = module
        self.integration_attempts = 0

    def __getattr__(self, name):
        return getattr(self.original, name)

    def _forbidden(self, *_args, **_kwargs):
        self.integration_attempts += 1
        raise AssertionError("static observability diagnostic forbids physics integration")

    mj_step = _forbidden
    mj_step1 = _forbidden
    mj_step2 = _forbidden


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("receipt.json"))
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or args.output.is_symlink():
        raise ValueError("static observability audit requires a new receipt path")
    root = Path(__file__).resolve().parents[3]
    assets = root.parent / "GR00T-WholeBodyControl"
    paths = {key: root / relative for key, (relative, _) in PINNED.items()}
    for key, path in paths.items():
        if sha(path) != PINNED[key][1]:
            raise ValueError(f"pinned {key} changed")
    manifest = json.loads(paths["manifest"].read_text())
    if manifest["source"]["checkpoint_sha256"] != PINNED["checkpoint"][1]:
        raise ValueError("pair is not bound to the tested v2 checkpoint")
    with np.load(paths["motion"], allow_pickle=False) as archive:
        motion = {key: archive[key].copy() for key in archive.files}
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    torch.set_num_threads(1)
    policy, policy_identity = load_generalist_pair(paths["manifest"], session_options=options)
    controller = CleanTrue23MujocoController(
        model_path=assets / MODEL, physics_path=root / PHYSICS, policy=policy
    )
    controller.module = NoIntegrationModule(controller.module)
    inputs = {str(path.resolve()): sha(path) for path in (
        *paths.values(), assets / MODEL, root / PHYSICS, Path(__file__),
        *(Path(value) for value in policy_identity["component_paths"].values()),
    )}
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("gear_sonic.") and path and Path(path).suffix == ".py":
            inputs[str(Path(path).resolve())] = sha(Path(path))

    packet = motion_reference_terms(motion, 9)
    joints = motion["joint_pos"][10]
    root_pos = motion["body_pos_w"][10, 0]
    root_quat = motion["body_quat_w"][10, 0]
    root_velocity = np.asarray([0.15, -0.1, 0.05, 0.12, -0.2, 0.07], dtype=np.float64)
    variants = (
        ("baseline", [0, 0, 0], [0, 0, 0]),
        ("robot_translated_8_minus3_m", [8, -3, 0], [0, 0, 0]),
        ("robot_linear_velocity_changed", [0, 0, 0], [1.5, -0.75, 0.2]),
    )
    rows, tensors = [], []
    for name, displacement, velocity_change in variants:
        velocity = root_velocity.copy()
        velocity[:3] += velocity_change
        controller.reset(
            base_position=root_pos + displacement,
            base_quaternion_wxyz=root_quat,
            joint_position_hardware=joints,
            root_velocity=velocity,
            joint_velocity_hardware=motion["joint_vel"][10],
            buffered_robot_pelvis_q9=motion["body_quat_w"][9, 0],
        )
        encoder = encoder267_from_reference(packet, controller.buffered_robot_pelvis_q9)
        frame = controller._policy_frame()
        history = term_major_history([frame.copy() for _ in range(10)])
        raw, decoder = policy.infer(encoder, history)
        values = dict(encoder267=encoder, history930=history, decoder994=decoder, raw23=raw)
        tensors.append(values)
        if controller.data.time != 0:
            raise AssertionError("static observation probe advanced simulation time")
        rows.append(dict(
            name=name,
            robot_displacement_xyz_m=displacement,
            root_linear_velocity_change_world_m_s=velocity_change,
            robot_root_xyz=controller.data.qpos[:3].tolist(),
            root_velocity_world_linear_then_body_angular=controller.data.qvel[:6].tolist(),
            root_error_from_source_frame10_m=(controller.data.qpos[:3] - root_pos).tolist(),
            simulation_time_s=float(controller.data.time),
            tensors={key: tensor_identity(value) for key, value in values.items()},
            raw_action23=raw.tolist(),
        ))
    for values in tensors[1:]:
        for key in tensors[0]:
            np.testing.assert_array_equal(tensors[0][key], values[key])

    command = SimpleNamespace(
        time_steps=torch.tensor([9]),
        motion=SimpleNamespace(
            joint_pos=torch.from_numpy(motion["joint_pos"]),
            time_step_total=len(motion["joint_pos"]),
        ),
        cfg=SimpleNamespace(body_names=tuple(controller.model.body(i).name for i in range(1, 25))),
        body_pos_w=torch.from_numpy(motion["body_pos_w"][9:10]),
        body_quat_w=torch.from_numpy(motion["body_quat_w"][9:10]),
        anchor_pos_w=torch.from_numpy(motion["body_pos_w"][9:10, 0]),
        anchor_quat_w=torch.from_numpy(motion["body_quat_w"][9:10, 0]),
        causal_robot_anchor_quat_w=torch.from_numpy(motion["body_quat_w"][9:10, 0]),
        robot_anchor_pos_w=torch.from_numpy(motion["body_pos_w"][10:11, 0]).clone(),
        robot_root_lin_vel_w=torch.from_numpy(root_velocity[None, :3].copy()).float(),
    )
    env = SimpleNamespace(num_envs=1, command_manager=SimpleNamespace(get_term=lambda _name: command))

    def training_features():
        return torch.cat((
            causal_history_lower_body(env),
            vr_3point_local_target(env, "motion"),
            vr_3point_local_orn_target(env, "motion"),
            causal_motion_anchor_ori_b(env),
        ), dim=-1)

    training_before = training_features()
    command.robot_anchor_pos_w += torch.tensor([[8, -3, 0]])
    command.robot_root_lin_vel_w += torch.tensor([[1.5, -0.75, 0.2]])
    training_after = training_features()
    if not torch.equal(training_before, training_after):
        raise AssertionError("training feature frontend is no longer translation invariant")
    if controller.module.integration_attempts:
        raise AssertionError("unexpected physics integration attempt")
    for path, digest in inputs.items():
        if sha(Path(path)) != digest:
            raise ValueError(f"source changed during static audit: {path}")

    receipt = dict(
        schema_version=1,
        kind="native23_static_root_observability_diagnostic_v1",
        runtime=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                     onnxruntime=ort.__version__, mujoco=controller.module.__version__,
                     inference_provider="CPUExecutionProvider"),
        fixture_scope="Three independent reset/FK-only observation fixtures; H10 repeats each static measured frame. Not physical motion replay, training, or qualification.",
        source_reference=dict(path=str(paths["motion"]), sha256=PINNED["motion"][1],
                              fixed_packet_q9=9, fixed_robot_pose_frame10=10,
                              reference_is_existing_fit_to_recorded_original29_policy_not_planned_choreography=True),
        states=rows,
        all_encoder_history_decoder_raw_exactly_equal=True,
        training_frontend=dict(
            scope="Actual causal/VR/orientation feature functions invoked on a minimal command fixture; no MJLab environment rollout.",
            before=tensor_identity(training_before.numpy()),
            after=tensor_identity(training_after.numpy()),
            exactly_equal=True,
        ),
        interpretation=[
            "The current 267+930 actor input cannot distinguish an arbitrary persistent XY displacement with identical orientation/joint/action histories and fixed source reference.",
            "Current actor does not directly observe actual or desired root linear velocity. This instantaneous invariance does not rule out indirect velocity inference from later proprioceptive/contact responses.",
            "267 contains reference leg history240, reference-pelvis-local point9 and orientation12 targets, and relative pelvis orientation6; 930 contains angular velocity, joints, actions and gravity histories.",
            "More optimization cannot remove the demonstrated position-offset alias without additional state/command information. Closed-loop source-world path/return qualification requires an approved root-feedback conditioning branch or a narrower root-relative qualification claim.",
            "This receipt establishes observation aliasing, not impossibility of performing a particular dance from a precisely matched initial state.",
        ],
        policy_identity=policy_identity,
        checkpoint_path=str(paths["checkpoint"]),
        input_files_sha256=inputs,
        physics_integration_steps=0,
        physics_integration_attempts=controller.module.integration_attempts,
        source_or_architecture_changed=False,
        training_launched=False,
        hardware_authorized=False,
        active_motor_control_authorized=False,
        deployment_ready=False,
        simulator_qualified=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(receipt=str(output), sha256=sha(output),
                          all_actor_inputs_and_outputs_exactly_equal=True,
                          training267_exactly_equal=True, physics_integration_steps=0)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
