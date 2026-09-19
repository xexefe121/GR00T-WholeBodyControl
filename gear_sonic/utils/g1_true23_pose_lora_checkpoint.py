"""Strict pose-LoRA original-intent snapshots for offline CPU research only.

Does not export a deployable pair or make old export/transport tools accept the
new actor. Checkpoint load remains weights-only with exact actor/lineage checks.
"""

from pathlib import Path

import numpy as np
import torch

from gear_sonic.envs.mjlab.sonic_true23_original_intent import RECIPE, objective_contract, validate_spec
from gear_sonic.trl.mjlab.native23_original_intent_actor import validate_original_intent_compatibility
from gear_sonic.trl.mjlab.native23_pose_lora_actor import (
    ACTOR_KIND,
    ARCHITECTURE,
    TRAINABLE,
    True23PoseLoraActorModel,
    pose_conditioned_decoder_forward,
)
from gear_sonic.trl.mjlab.native23_pose_lora_runner import (
    CHECKPOINT_HEADER,
    MULTIPLIERS,
    PROFILE,
    training_contract,
    validate_pose_lora_checkpoint,
)
from gear_sonic.utils.g1_23dof_contract import LOW_LATENCY_RELEASE_SHA256
from gear_sonic.utils.g1_23dof_mjlab_training import validate_mjlab_training_lineage
from gear_sonic.utils.g1_true23_buffered_reference import BUFFERED_TIMING, reference_profile_contract
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_original_intent_checkpoint import OriginalIntentCPUActor
from gear_sonic.utils.g1_true23_root_feedback import root_feedback_contract

ROOT = Path(__file__).resolve().parents[2]


def validate_semantics(checkpoint):
    if not isinstance(checkpoint, dict) or checkpoint.get("header") != CHECKPOINT_HEADER:
        raise ValueError("original-intent reader requires research checkpoint header")
    contract = checkpoint.get("actor", {}).get("contract", {})
    if contract.get("kind") != ACTOR_KIND:
        raise ValueError("original-intent reader rejects old/relabelled actor kinds")
    compatibility = validate_original_intent_compatibility(contract.get("release_compatibility"))
    lineage = validate_mjlab_training_lineage(checkpoint.get("lineage"))
    if checkpoint.get("lineage_sha256") != lineage["lineage_sha256"]:
        raise ValueError("original-intent checkpoint lineage differs")
    resolved = lineage["materials"]["resolved_config"]["payload"]
    generalist = resolved.get("native23_generalist", {})
    feedback = resolved.get("native23_root_feedback", {})
    agent = resolved.get("agent", {})
    if (
        resolved.get("semantic_profile") != reference_profile_contract(BUFFERED_TIMING)
        or resolved.get("action_count") != 23
        or resolved.get("policy_dim") != 930
        or resolved.get("tokenizer_dim") != 268
        or generalist.get("source_checkpoint_sha256") != LOW_LATENCY_RELEASE_SHA256
        or generalist.get("encoder_and_fsq_frozen") is not True
        or generalist.get("decoder_frozen") is not True
        or generalist.get("trainable") != TRAINABLE
        or resolved.get("native23_pose_lora") != training_contract()
        or feedback.get("trainable") != TRAINABLE
        or feedback.get("optimizer_profile") != PROFILE
        or feedback.get("optimizer_learning_rate_multipliers") != MULTIPLIERS
        or feedback.get("pose_lora_training") != training_contract()
        or generalist.get("deployment_ready") is not False
        or feedback.get("feature_contract") != root_feedback_contract(BUFFERED_TIMING)
        or feedback.get("reference_timing") != BUFFERED_TIMING
        or feedback.get("architecture") != ARCHITECTURE
        or feedback.get("release_compatibility") != compatibility
        or feedback.get("objective_profile") != RECIPE
        or feedback.get("objective_profile_contract") != objective_contract()
        or feedback.get("old_live_or_export_compatibility_claimed") is not False
        or feedback.get("same_objective_or_reference_resume_claimed") is not False
        or feedback.get("critic_observes_original_q1_vr21") is not True
        or feedback.get("deployment_ready") is not False
        or feedback.get("hardware_authorized") is not False
        or feedback.get("continuation") is not None
        or feedback.get("curriculum", {}).get("parent_actor_initialization") is not None
        or "ppo_auxiliary_objective" in feedback
        or agent.get("obs_groups")
        != {
            "actor": ["tokenizer", "policy", "root_feedback"],
            "critic": ["critic", "root_feedback", "original_intent_value_reference"],
        }
        or agent.get("actor", {}).get("class_name")
        != "gear_sonic.trl.mjlab.native23_pose_lora_actor:True23PoseLoraActorModel"
        or agent.get("actor", {}).get("release_compatibility") != compatibility
    ):
        raise ValueError("original-intent executed training semantics differ")
    spec = validate_spec(feedback.get("original_intent_spec"))
    for name, key in (("source_model", "source_geometry_sha256"), ("native_model", "native_geometry_sha256")):
        if spec["files"][name]["sha256"] != compatibility[key]:
            raise ValueError("original-intent reference and actor geometry differ")
    # Verify the bound repository source closure, not just its declared hash.
    rows = lineage["materials"]["source_files"]["files"]
    reverified = {}
    for row in rows:
        name = row["logical_path"]
        if not name.startswith("gear_sonic/"):
            continue
        path = (ROOT / name).resolve(strict=True)
        if (
            not path.is_relative_to(ROOT)
            or sha256_file(path) != row["sha256"]
            or path.stat().st_size != row["size_bytes"]
        ):
            raise ValueError(f"original-intent executed source changed: {name}")
        reverified[str(path)] = row["sha256"]
    required = {
        "gear_sonic/envs/mjlab/sonic_true23_original_intent.py",
        "gear_sonic/trl/mjlab/native23_original_intent_actor.py",
        "gear_sonic/scripts/train_g1_true23_original_intent.py",
        "gear_sonic/scripts/train_g1_true23_pose_lora.py",
        "gear_sonic/trl/mjlab/native23_pose_lora_actor.py",
        "gear_sonic/trl/mjlab/native23_pose_lora_runner.py",
    }
    if not required.issubset({row["logical_path"] for row in rows}):
        raise ValueError("original-intent implementation missing from executed source closure")
    original_row = next(
        (r for r in rows if r["logical_path"] == "native23_original_intent/original_reference.npz"), None
    )
    if original_row is None or original_row["sha256"] != spec["files"]["original_reference"]["sha256"]:
        raise ValueError("original reference is not bound by executed lineage")
    motion_rows = lineage["materials"]["motion_dataset"]["files"]
    if not any(r["sha256"] == spec["files"]["native_motion"]["sha256"] for r in motion_rows):
        raise ValueError("paired native motion is not bound by executed lineage")
    critic = checkpoint.get("critic_state_dict", {})
    if critic.get("mlp.0.weight", torch.empty(0)).shape != (512, 286):
        raise ValueError("original-intent critic must consume256+9+21 actual inputs")
    return dict(
        resolved=resolved, spec=spec, compatibility=compatibility, reverified_repository_sources=reverified
    )


def construct_actor(checkpoint, semantics, *, warm_start_path, source_checkpoint_path):
    exploration = checkpoint["actor"]["contract"]["exploration"]
    return True23PoseLoraActorModel(
        {"tokenizer": torch.zeros(1, 267), "policy": torch.zeros(1, 930), "root_feedback": torch.zeros(1, 9)},
        {"actor": ["tokenizer", "policy", "root_feedback"]},
        "actor",
        23,
        warm_start_path=str(warm_start_path),
        source_checkpoint_path=str(source_checkpoint_path),
        release_compatibility=semantics["compatibility"],
        std_min=exploration["std_min"],
        std_max=exploration["std_max"],
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "std_type": "scalar",
            "init_std": exploration["init_std"],
        },
    )


class PoseLoraCPUActor(OriginalIntentCPUActor):
    """Keep the trace ABI, but execute the pose branch as well as root feedback."""

    def __init__(self, actor):
        if not isinstance(actor, True23PoseLoraActorModel):
            raise TypeError("pose-LoRA CPU reader requires its distinct actor")
        super().__init__(actor)

    def infer(self, encoder267, history930, root_feedback9):
        for value, width in ((encoder267, 267), (history930, 930), (root_feedback9, 9)):
            if (
                not isinstance(value, np.ndarray)
                or value.shape != (width,)
                or value.dtype != np.float32
                or not np.isfinite(value).all()
            ):
                raise ValueError("CPU pose-LoRA inputs require finite float32 267/930/9 arrays")
        with torch.inference_mode():
            token = self.actor.core.encode(torch.from_numpy(encoder267.copy())[None])
            history = torch.from_numpy(history930.copy())[None]
            raw_combined = torch.cat((token, history), -1)
            combined = torch.cat((token, self.actor.core.codec.encode_proprioception(history)), -1)
            output = pose_conditioned_decoder_forward(
                self.actor, combined, torch.from_numpy(root_feedback9.copy())[None]
            )
        if output.shape != (1, 23) or not torch.isfinite(output).all():
            raise ValueError("CPU pose-LoRA actor produced invalid action")
        return output.numpy()[0].copy(), raw_combined.numpy()[0].copy()


def load_cpu_actor(checkpoint_path, *, warm_start_path, source_checkpoint_path):
    requested = Path(checkpoint_path)
    if requested.is_symlink():
        raise ValueError("checkpoint may not be a symlink")
    path = requested.resolve(strict=True)
    digest = sha256_file(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    semantics = validate_semantics(checkpoint)
    actor = construct_actor(
        checkpoint, semantics, warm_start_path=warm_start_path, source_checkpoint_path=source_checkpoint_path
    )
    validate_pose_lora_checkpoint(checkpoint, actor=actor, lineage=checkpoint["lineage"])
    actor.load_training_artifact(checkpoint["actor"])
    if sha256_file(path) != digest:
        raise ValueError("checkpoint changed during read")
    identity = dict(
        kind="native23_pose_lora_original_intent_pytorch_cpu_research_reader_v1",
        checkpoint_path=str(path),
        checkpoint_sha256=digest,
        actor_state_sha256=checkpoint["actor"]["state_sha256"],
        lineage_sha256=checkpoint["lineage_sha256"],
        completed_updates=checkpoint["trainer_state"]["completed_update_count"],
        release_compatibility=semantics["compatibility"],
        backend="pytorch_cpu_deterministic_mean_not_ONNX",
        export_or_transport_acceptance_claimed=False,
        deployment_ready=False,
        hardware_authorized=False,
    )
    return PoseLoraCPUActor(actor), identity, semantics
