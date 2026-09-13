"""Diagnostic two-input native23 root-feedback pair, never hardware promotion."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import copy
import gc
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from gear_sonic.scripts.export_g1_true23_generalist import EncoderExport
from gear_sonic.trl.mjlab.native23_root_feedback_actor import (
    ROOT_FEEDBACK_ARCHITECTURE,
    True23RootFeedbackActorModel,
    conditioned_decoder_forward,
)
from gear_sonic.trl.mjlab.native23_root_feedback_runner import CHECKPOINT_HEADER, validate_root_feedback_checkpoint
from gear_sonic.utils import g1_23dof_artifact as artifact
from gear_sonic.utils.g1_23dof_contract import LOW_LATENCY_RELEASE_SHA256
from gear_sonic.utils.g1_23dof_mjlab_training import validate_mjlab_training_lineage
from gear_sonic.utils.g1_true23_buffered_reference import reference_profile_contract
from gear_sonic.utils.g1_true23_root_feedback import root_feedback_contract


class RootFeedbackDecoderExport(nn.Module):
    def __init__(self, decoder, conditioner, proprioception_mask):
        super().__init__()
        self.decoder = decoder
        self.conditioner = conditioner
        self.register_buffer("input_keep_mask", torch.cat((torch.ones(64), proprioception_mask.detach().cpu())))

    def forward(self, obs_dict, root_feedback):
        return conditioned_decoder_forward(
            self.decoder, self.conditioner, obs_dict * self.input_keep_mask, root_feedback
        )


def validate_export_semantics(checkpoint):
    if not isinstance(checkpoint, Mapping) or checkpoint.get("header") != CHECKPOINT_HEADER:
        raise ValueError("root-feedback export requires distinct versioned checkpoint header")
    lineage = validate_mjlab_training_lineage(checkpoint.get("lineage"))
    if checkpoint.get("lineage_sha256") != lineage["lineage_sha256"]:
        raise ValueError("root-feedback export lineage hash mismatch")
    resolved = lineage["materials"]["resolved_config"]["payload"]
    generalist, feedback = resolved.get("native23_generalist"), resolved.get("native23_root_feedback")
    if not isinstance(feedback, Mapping):
        raise ValueError("root-feedback export requires training configuration")
    if "ppo_auxiliary_objective" in feedback:
        from gear_sonic.trl.mjlab.native23_projected_target_ppo import projection_objective_contract

        objective = feedback["ppo_auxiliary_objective"]
        if not isinstance(objective, Mapping) or objective != projection_objective_contract(objective.get("name")):
            raise ValueError("root-feedback export has mismatched PPO auxiliary objective")
    timing = feedback.get("reference_timing", "causal_history")
    if resolved.get("semantic_profile") != reference_profile_contract(timing):
        raise ValueError("root-feedback export requires exact executed SONIC semantic contract")
    if not isinstance(generalist, Mapping) or (
        generalist.get("source_checkpoint_sha256") != LOW_LATENCY_RELEASE_SHA256
        or generalist.get("encoder_and_fsq_frozen") is not True
        or generalist.get("deployment_ready") is not False
    ):
        raise ValueError("root-feedback export requires pinned frozen SONIC source lineage")
    if not isinstance(feedback, Mapping) or (
        feedback.get("feature_contract") != root_feedback_contract(timing)
        or feedback.get("architecture") != ROOT_FEEDBACK_ARCHITECTURE
        or feedback.get("deployment_ready") is not False
    ):
        raise ValueError("root-feedback export requires exact distinct root feature contract")
    compatibility = feedback.get("release_compatibility")
    if compatibility is not None:
        from gear_sonic.utils.g1_true23_release_compatibility import validate_release_compatibility

        validate_release_compatibility(compatibility)
        if compatibility.get("reference_timing", "causal_history") != timing:
            raise ValueError("release compatibility and executed reference timing differ")
        if checkpoint["actor"]["contract"].get("release_compatibility") != compatibility:
            raise ValueError("actor and executed training release compatibility differ")
    elif checkpoint.get("actor", {}).get("contract", {}).get("release_compatibility") is not None:
        raise ValueError("actor release compatibility lacks executed training evidence")
    if timing != "causal_history" and compatibility is None:
        raise ValueError("buffered reference requires explicit versioned release compatibility")
    return {
        **({"release_compatibility": copy.deepcopy(compatibility)} if compatibility is not None else {}),
        "semantic_profile": copy.deepcopy(resolved["semantic_profile"]),
        "root_feedback_contract": root_feedback_contract(timing),
        "training_configuration": copy.deepcopy(generalist),
        "root_feedback_training_configuration": copy.deepcopy(feedback),
    }


def validate_root_decoder_onnx(model):
    import onnx

    onnx.checker.check_model(model, full_check=True)
    if [value.version for value in model.opset_import if value.domain in ("", "ai.onnx")] != [
        artifact.ONNX_OPSET_VERSION
    ]:
        raise ValueError("root-feedback ONNX opset mismatch")
    expected = [("obs_dict", [1, 994]), ("root_feedback", [1, 9])]
    actual = [
        (value.name, [dim.dim_value for dim in value.type.tensor_type.shape.dim]) for value in model.graph.input
    ]
    if actual != expected or any(
        value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT for value in model.graph.input
    ):
        raise ValueError("root-feedback decoder requires separate float32 [1,994] and [1,9] inputs")
    outputs = [
        (value.name, [dim.dim_value for dim in value.type.tensor_type.shape.dim]) for value in model.graph.output
    ]
    if (
        outputs != [("action", [1, 23])]
        or model.graph.output[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT
    ):
        raise ValueError("root-feedback decoder requires exactly 23 physical float32 outputs")
    if any(
        value.data_location == onnx.TensorProto.EXTERNAL or value.external_data
        for value in model.graph.initializer
    ):
        raise ValueError("root-feedback ONNX may not load external initializer files")
    # Both declared inputs must actually reach the output graph. A stale old
    # decoder with a merely declared extra input is not the new architecture.
    reachable = {"root_feedback"}
    for node in model.graph.node:
        if any(name in reachable for name in node.input):
            reachable.update(node.output)
    if "action" not in reachable:
        raise ValueError("root-feedback decoder output does not depend on its root input graph")


def validate_root_decoder_parity(model, path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(2309)
    probes = [
        (np.zeros((1, 994), np.float32), np.zeros((1, 9), np.float32)),
        (
            rng.normal(0, 0.1, (1, 994)).astype(np.float32),
            np.array([[1.0, -2.0, 0.1, 0.2, 0.0, 0.0, 0.0, 1.0, 0.3]], np.float32),
        ),
        (rng.normal(0, 0.2, (1, 994)).astype(np.float32), rng.normal(0, 1, (1, 9)).astype(np.float32)),
    ]
    maximum = 0.0
    with torch.no_grad():
        for original, feedback in probes:
            expected = model(torch.from_numpy(original), torch.from_numpy(feedback)).cpu().numpy()
            actual = session.run(["action"], {"obs_dict": original, "root_feedback": feedback})[0]
            if actual.shape != (1, 23) or not np.isfinite(actual).all():
                raise ValueError("root-feedback ONNX output shape/finiteness mismatch")
            maximum = max(maximum, float(np.max(np.abs(actual - expected))))
            if not np.allclose(actual, expected, atol=1e-5, rtol=1e-4):
                raise ValueError(f"root-feedback decoder ONNX parity failed: {maximum}")
    return {
        "probe_count": len(probes),
        "includes_nonzero_root_feedback": True,
        "parity_max_abs_error": maximum,
        "atol": 1e-5,
        "rtol": 1e-4,
        "parity_passed": True,
    }


def export_pair(*, checkpoint_path, warm_start_path, source_checkpoint_path, output_directory):
    import onnx

    output, checkpoint_file = Path(output_directory).expanduser(), Path(checkpoint_path).expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"root-feedback diagnostic output directory already exists: {output}")
    if checkpoint_file.is_symlink():
        raise ValueError("root-feedback export checkpoint may not be a symlink")
    checkpoint_file = checkpoint_file.resolve(strict=True)
    checkpoint_hash = artifact.sha256_file(checkpoint_file)
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    semantics = validate_export_semantics(checkpoint)
    exploration = checkpoint["actor"]["contract"]["exploration"]
    actor = True23RootFeedbackActorModel(
        {"tokenizer": torch.zeros(1, 267), "policy": torch.zeros(1, 930), "root_feedback": torch.zeros(1, 9)},
        {"actor": ["tokenizer", "policy", "root_feedback"]},
        "actor",
        23,
        warm_start_path=str(warm_start_path),
        source_checkpoint_path=str(source_checkpoint_path),
        release_compatibility=semantics.get("release_compatibility"),
        std_min=exploration["std_min"],
        std_max=exploration["std_max"],
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "std_type": "scalar",
            "init_std": exploration["init_std"],
        },
    )
    validate_root_feedback_checkpoint(checkpoint, actor=actor, lineage=checkpoint["lineage"])
    actor.load_training_artifact(checkpoint["actor"])
    actor.eval()
    source = {
        "checkpoint_filename": checkpoint_file.name,
        "checkpoint_sha256": checkpoint_hash,
        "actor_state_sha256": checkpoint["actor"]["state_sha256"],
        "lineage_sha256": checkpoint["lineage_sha256"],
        "completed_update_count": checkpoint["trainer_state"]["completed_update_count"],
        "architecture_initialization_profile": actor.reference_profile,
        "paired_release_checkpoint_sha256": actor.core.source_checkpoint_sha256,
        "frozen_encoder_state_sha256": actor.core.frozen_encoder_sha256(),
    }
    del checkpoint
    gc.collect()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 2,
        "kind": "g1_native23_root_feedback_diagnostic_pair",
        "source": source,
        **semantics,
        "actor_contract": actor.artifact_contract(),
        "proprioception_absent_slots_fixed_zero_in_decoder_graph": True,
        "diagnostic_only": True,
        "deployment_ready": False,
        "promotion_eligible": False,
        "hardware_authorized": False,
        "active_motor_control_authorized": False,
        "completed_motion_qualification": False,
        "physical_root_state_estimator_qualified": False,
    }
    models = {
        "encoder": EncoderExport(actor.core.encoder).eval(),
        "decoder": RootFeedbackDecoderExport(
            actor.core.decoder, actor.root_conditioner, actor.core.codec.proprioception_keep_mask
        ).eval(),
    }
    for name, model in models.items():
        path = output / f"{name}.root_feedback.diagnostic.onnx"
        encoder = name == "encoder"
        inputs = torch.zeros(1, 267) if encoder else (torch.zeros(1, 994), torch.zeros(1, 9))
        input_names, output_name = (
            (["teleop_obs"], "token") if encoder else (["obs_dict", "root_feedback"], "action")
        )
        with torch.no_grad():
            torch.onnx.export(
                model,
                inputs,
                path,
                input_names=input_names,
                output_names=[output_name],
                opset_version=artifact.ONNX_OPSET_VERSION,
                dynamo=False,
                export_params=True,
                do_constant_folding=True,
                dynamic_axes=None,
            )
        graph = onnx.shape_inference.infer_shapes(
            onnx.load(path, load_external_data=False), strict_mode=True, data_prop=True
        )
        onnx.helper.set_model_props(
            graph,
            {
                **(
                    {"release_compatibility_sha256": semantics["release_compatibility"]["contract_sha256"]}
                    if "release_compatibility" in semantics
                    else {}
                ),
                "artifact_role": f"native23_root_feedback_diagnostic_{name}",
                "deployment_ready": "false",
                "hardware_authorized": "false",
                "source_checkpoint_sha256": checkpoint_hash,
                "actor_state_sha256": source["actor_state_sha256"],
                "semantic_profile": semantics["semantic_profile"]["profile"],
                "semantic_contract_sha256": semantics["semantic_profile"]["contract_sha256"],
                "root_feedback_contract_sha256": semantics["root_feedback_contract"]["contract_sha256"],
                "architecture_initialization_profile": actor.reference_profile,
            },
        )
        onnx.save_model(graph, path, save_as_external_data=False)
        if encoder:
            artifact.validate_encoder_onnx_structure(graph)
            parity = artifact.validate_ort_parity(
                model, path, input_name="teleop_obs", input_dim=267, output_name="token", output_dim=64
            )
            if parity["parity_max_abs_error"] != 0:
                raise ValueError("root-feedback frozen SONIC encoder requires exact token parity")
            signature = dict(
                input_name="teleop_obs", input_shape=[1, 267], output_name="token", output_shape=[1, 64]
            )
        else:
            validate_root_decoder_onnx(graph)
            parity = validate_root_decoder_parity(model, path)
            signature = dict(
                inputs=[{"name": "obs_dict", "shape": [1, 994]}, {"name": "root_feedback", "shape": [1, 9]}],
                output_name="action",
                output_shape=[1, 23],
            )
        report[name] = {"filename": path.name, "sha256": artifact.sha256_file(path), **signature, "parity": parity}
    if artifact.sha256_file(checkpoint_file) != checkpoint_hash:
        raise ValueError("root-feedback source checkpoint changed during export")
    with (output / "root_feedback.diagnostic.json").open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "warm-start", "source-checkpoint", "output-directory"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            export_pair(
                checkpoint_path=args.checkpoint,
                warm_start_path=args.warm_start,
                source_checkpoint_path=args.source_checkpoint,
                output_directory=args.output_directory,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
