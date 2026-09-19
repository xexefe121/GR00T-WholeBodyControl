"""Saved-input compact tracker referee adapter; never a robot controller."""

from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from gear_sonic.trl.mjlab.native23_compact_actor_v2 import CompactNative23ActorV2
from gear_sonic.utils.g1_23dof_contract import REFERENCE_PROFILE_NORMAL
from gear_sonic.utils.g1_true23_compact_features import contract, make_goal, pack_observation
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_released_core_comparison import ReleasedCoreAdapter
from gear_sonic.utils.g1_true23_root_feedback import root_feedback_numpy


class CompactPolicy:
    profile = REFERENCE_PROFILE_NORMAL

    def __init__(self, checkpoint):
        self.path = Path(checkpoint).resolve(strict=True)
        self.sha256 = sha256_file(self.path)
        saved = torch.load(self.path, map_location="cpu", weights_only=True)
        if saved.get("kind") != "native23_compact_reference_tracker_training_v1":
            raise ValueError("not a compact native23 checkpoint")
        self.manifest = saved["manifest"]
        if self.manifest["contract"] != contract() or saved.get("deployment_ready") is not False:
            raise ValueError("compact snapshot contract mismatch")
        for path, expected in self.manifest["source_hashes"].items():
            if sha256_file(Path(path)) != expected:
                raise ValueError("compact source changed: " + path)
        self.actor = CompactNative23ActorV2(
            {"native_controller": torch.zeros(1, 165)}, {"actor": ["native_controller"]}, "actor", 23
        ).eval()
        self.actor.load_state_dict(saved["actor_state"], strict=True)
        self.actor.requires_grad_(False)
        self.completed_updates = saved["completed_updates"]

    def predict(self, features):
        if features.shape != (165,) or features.dtype != np.float32 or not np.isfinite(features).all():
            raise ValueError("compact replay requires finite float32 observation165")
        with torch.inference_mode():
            return self.actor({"native_controller": torch.from_numpy(features[None])})[0].numpy().copy()

    def identity(self):
        if sha256_file(self.path) != self.sha256:
            raise ValueError("checkpoint changed during replay")
        return dict(
            kind="compact_native23_cpu_referee_v1",
            checkpoint_sha256=self.sha256,
            completed_training_updates=self.completed_updates,
            actor_contract=contract(),
            frozen_sonic_policy_parity_claimed=False,
            deployment_ready=False,
            hardware_authorized=False,
        )


class CompactAdapter(ReleasedCoreAdapter):
    def __init__(self, motion, virtual_vr, *, root, assets):
        super().__init__(motion, virtual_vr, profile=REFERENCE_PROFILE_NORMAL, root=root, assets=assets)
        self.geometry = mujoco.MjModel.from_xml_path(str(Path(assets) / MODEL))
        self.measurement_fk = mujoco.MjData(self.geometry)
        self.reference_fk = mujoco.MjData(self.geometry)
        self.feet = [self.geometry.body(name).id for name in ("left_ankle_roll_link", "right_ankle_roll_link")]

    def goal_from_copies(self, control_index, qpos, qvel):
        q0, q1 = control_index + 9, control_index + 10
        if q1 >= self.emission_start + control_index:
            raise ValueError("compact goal may not consume an unreceived sample")
        source = self.source
        desired = source["root_position_w"][q1]
        previous = source["root_position_w"][q0]
        self.measurement_fk.qpos[:] = qpos
        self.reference_fk.qpos[:] = np.r_[desired, source["root_quaternion_wxyz"][q1], source["joint_pos"][q1]]
        mujoco.mj_kinematics(self.geometry, self.measurement_fk)
        mujoco.mj_kinematics(self.geometry, self.reference_fk)
        error = (self.reference_fk.xpos[self.feet] - self.measurement_fk.xpos[self.feet]).astype(np.float32)
        feedback = root_feedback_numpy(
            desired, qpos[:3], (desired - previous) / np.float32(0.02), qvel[:3], qpos[3:7]
        )
        arrays = (
            source["joint_pos"][q1],
            (source["joint_pos"][q1] - source["joint_pos"][q0]) / np.float32(0.02),
            feedback,
            source["root_quaternion_wxyz"][q1],
            qpos[3:7].astype(np.float32),
            source["virtual_vr21"][q1],
            error,
            np.array([desired[2], qpos[2]], np.float32),
        )
        with torch.inference_mode():
            return make_goal(*(torch.from_numpy(value[None]) for value in arrays))[0].numpy().copy()

    def infer(self, policy, encoder, history, *, control_index, **state):
        if type(policy) is not CompactPolicy:
            raise ValueError("compact adapter refuses another actor family")
        goal = self.goal_from_copies(control_index, state["measured_qpos"], state["measured_qvel"])
        with torch.inference_mode():
            features = (
                pack_observation(torch.from_numpy(history[None]), torch.from_numpy(goal[None]))[0].numpy().copy()
            )
        # The legacy referee requires994 fields. These zero64 values are an
        # explicitly labelled non-neural compatibility stub, NOT SONIC tokens.
        stub = np.r_[np.zeros(64, np.float32), history].astype(np.float32)
        proxy = SimpleNamespace(profile=self.profile, infer=lambda e, h: (policy.predict(features), stub.copy()))
        before = len(self.attempts)
        try:
            return super().infer(proxy, encoder, history, control_index=control_index, **state)
        finally:
            if len(self.attempts) == before + 1:
                self.attempts[-1].update(
                    native_goal90=goal,
                    native_controller165=features,
                    source_goal_index=np.int64(control_index + 10),
                )

    def contract(self):
        return {
            **super().contract(),
            "kind": "compact_native23_guarded_cpu_adapter_v1",
            "decoder994_is_non_neural_referee_stub": True,
            "actor_is_frozen_sonic": False,
            "current_goal_features": contract(),
            "root_feedback_used": True,
            "physical_estimator_qualified": False,
        }
