"""Separately versioned original-core Root9 CPU diagnostic, never robot control."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from gear_sonic.trl.mjlab.frozen_platform_lora_actor import _tensor_state_sha256
from gear_sonic.trl.mjlab.native23_normal_lora_actor import True23NormalLoraActorModel
from gear_sonic.trl.mjlab.native23_normal_lora_runner import validate_normal_checkpoint
from gear_sonic.utils.g1_23dof_contract import REFERENCE_PROFILE_NORMAL
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_normal_reference import normal_reference_contract, normal_root_feedback_contract
from gear_sonic.utils.g1_true23_released_core_comparison import ReleasedCoreAdapter
from gear_sonic.utils.g1_true23_root_feedback import root_feedback_numpy


class NormalLoraPolicy:
    """Load only a hash-bound normal snapshot; preserve original267/994 interfaces."""

    profile = REFERENCE_PROFILE_NORMAL

    def __init__(self, checkpoint_path, *, warm_start_path, source_checkpoint_path):
        self.path = Path(checkpoint_path).resolve(strict=True)
        checkpoint = torch.load(self.path, map_location="cpu", weights_only=True)
        obs = {
            key: torch.zeros(1, width)
            for key, width in (("tokenizer", 267), ("policy", 930), ("root_feedback", 9))
        }
        self.actor = True23NormalLoraActorModel(
            obs,
            {"actor": ("tokenizer", "policy", "root_feedback")},
            "actor",
            23,
            warm_start_path=warm_start_path,
            source_checkpoint_path=source_checkpoint_path,
            distribution_cfg=dict(class_name="GaussianDistribution", init_std=0.1, std_type="scalar"),
        ).eval()
        self.lineage = checkpoint["lineage"]
        validate_normal_checkpoint(checkpoint, actor=self.actor, lineage=self.lineage)
        self.actor.load_training_artifact(checkpoint["actor"])
        self.completed_updates = checkpoint["trainer_state"]["completed_update_count"]
        self.initial_actor_sha256 = checkpoint["actor"]["state_sha256"]
        self.checkpoint_sha256 = sha256_file(self.path)
        self.reverified_sources = {}
        root = Path(__file__).resolve().parents[2]
        for entry in self.lineage["materials"]["source_files"]["files"]:
            logical = entry["logical_path"]
            if logical.startswith("gear_sonic/"):
                path = (root / logical).resolve(strict=True)
                if (
                    not path.is_relative_to(root)
                    or sha256_file(path) != entry["sha256"]
                    or path.stat().st_size != entry["size_bytes"]
                ):
                    raise ValueError("normal replay repository source differs from training: " + logical)
                self.reverified_sources[str(path)] = entry["sha256"]

    def infer(self, encoder267, history930, root_feedback9):
        for value, width in ((encoder267, 267), (history930, 930), (root_feedback9, 9)):
            if value.shape != (width,) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError("normal replay requires finite float32267/930/9 vectors")
        with torch.inference_mode():
            history = torch.from_numpy(history930[None])
            self.actor.core.codec.validate_padded_proprioception(history)
            encoder = torch.from_numpy(encoder267[None])
            root = torch.from_numpy(root_feedback9[None])
            raw = self.actor(dict(tokenizer=encoder, policy=history, root_feedback=root))
            decoder = torch.cat((self.actor.core.encode(encoder), history), -1)
        if not torch.isfinite(raw).all():
            raise ValueError("normal replay emitted a nonfinite action")
        return raw[0].numpy().copy(), decoder[0].numpy().copy()

    def identity(self):
        self.actor.core.assert_frozen_encoder_unchanged()
        actual = _tensor_state_sha256(self.actor.state_dict())
        if actual != self.initial_actor_sha256 or sha256_file(self.path) != self.checkpoint_sha256:
            raise ValueError("normal snapshot or actor changed during replay")
        return dict(
            kind="native23_normal_lora_root9_cpu_diagnostic_identity_v1",
            checkpoint_sha256=self.checkpoint_sha256,
            actor_state_sha256=actual,
            completed_training_updates=self.completed_updates,
            actor_contract=self.actor.artifact_contract(),
            reference=normal_reference_contract(),
            root_feedback=normal_root_feedback_contract(),
            hardware_authorized=False,
            deployment_ready=False,
        )


class NormalLoraAdapter(ReleasedCoreAdapter):
    """Reuse unchanged920ms admission and guard; add Root9 from copied current state."""

    def __init__(self, motion, virtual_vr, *, root, assets):
        super().__init__(motion, virtual_vr, profile=REFERENCE_PROFILE_NORMAL, root=root, assets=assets)

    def infer(self, policy, encoder, history, *, control_index, **state):
        if type(policy) is not NormalLoraPolicy:
            raise ValueError("normal adapter refuses another checkpoint family")
        desired = state["desired_position_w"].astype(np.float32)
        previous = state["previous_desired_position_w"].astype(np.float32)
        qpos, qvel = state["measured_qpos"], state["measured_qvel"]
        if qpos.shape != (30,) or qvel.shape != (29,):
            raise ValueError("normal adapter requires copied native23 free-joint state")
        feedback = root_feedback_numpy(
            desired,
            qpos[:3].astype(np.float32),
            (desired - previous) / np.float32(0.02),
            qvel[:3].astype(np.float32),
            qpos[3:7].astype(np.float32),
        )
        proxy = SimpleNamespace(profile=policy.profile, infer=lambda e, h: policy.infer(e, h, feedback.copy()))
        count = len(self.attempts)
        try:
            return super().infer(proxy, encoder, history, control_index=control_index, **state)
        finally:
            if len(self.attempts) == count + 1:
                self.attempts[-1].update(
                    root_feedback9=feedback.copy(),
                    desired_root_position_w=desired.copy(),
                    previous_desired_root_position_w=previous.copy(),
                )

    def contract(self):
        return {
            **super().contract(),
            "kind": "native23_normal_lora_root9_guarded_cpu_adapter_v1",
            "root_feedback_used": True,
            "root_feedback": normal_root_feedback_contract(),
            "reference": normal_reference_contract(),
            "measured_state_source": "privileged_simulator_ground_truth",
            "physical_estimator_qualified": False,
            "checkpoint_export_for_hardware": False,
        }
