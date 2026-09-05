"""Materialize merged true23 weights for simulator evaluation only."""

from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib
import os
from pathlib import Path
import re
import tempfile
from typing import Any
import warnings

import torch

from gear_sonic.trl.mjlab.frozen_platform_lora_actor import (
    FrozenPlatformTrue23Core,
)
from gear_sonic.trl.mjlab.frozen_platform_lora_runner import (
    CHECKPOINT_HEADER,
    load_frozen_platform_lora_checkpoint,
)
from gear_sonic.utils.g1_23dof_artifact import (
    canonical_json_bytes,
    inspect_true23_policy_state,
    sha256_file,
)
from gear_sonic.utils.g1_23dof_contract import (
    DEPLOYMENT_DECODER_INPUT_DIM,
    DEPLOYMENT_HISTORY_LENGTH,
    OBS_LAYOUT_PADDED_IL29,
    TARGET_DOF,
)
from gear_sonic.utils.g1_23dof_mjlab_training import (
    validate_mjlab_training_lineage,
)

DIAGNOSTIC_HEADER = "g1_true23_frozen_lora_diagnostic_policy"
DIAGNOSTIC_KIND = "g1_true23_frozen_lora_merged_diagnostic_policy"
DIAGNOSTIC_SCHEMA_VERSION = 1


def _header() -> dict[str, Any]:
    return {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "kind": DIAGNOSTIC_KIND,
        "role": "simulator_evaluation_only",
        "deployment_ready": False,
        "promotion_eligible": False,
        "hardware_authorized": False,
    }


def _sha256_mapping(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(value))).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)
    )


def _copy_policy(value: Any) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("diagnostic policy_state_dict must be a non-empty mapping")
    result: dict[str, torch.Tensor] = {}
    for name, tensor in value.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("diagnostic policy must map names to tensors")
        copied = tensor.detach().cpu().float().contiguous().clone()
        if not torch.isfinite(copied).all():
            raise ValueError(f"diagnostic policy contains NaN or Inf: {name}")
        result[name] = copied
    return result


def validate_frozen_lora_diagnostic_policy(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("frozen LoRA diagnostic artifact must be a mapping")
    expected = {
        DIAGNOSTIC_HEADER,
        "policy_state_dict",
        "policy_state_sha256",
        "g1_23dof_metadata",
        "adapter_contract",
        "adapter_contract_sha256",
        "adapter_state_sha256",
        "update_count",
        "training_lineage",
        "training_lineage_sha256",
        "source_resume",
    }
    if set(value) != expected:
        raise ValueError("frozen LoRA diagnostic artifact root key mismatch")
    if value.get(DIAGNOSTIC_HEADER) != _header():
        raise ValueError("frozen LoRA diagnostic header mismatch")
    contract = value.get("adapter_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("frozen LoRA diagnostic adapter contract is missing")
    if value.get("adapter_contract_sha256") != _sha256_mapping(contract):
        raise ValueError("frozen LoRA diagnostic adapter contract hash mismatch")
    lineage = validate_mjlab_training_lineage(value.get("training_lineage"))
    if value.get("training_lineage_sha256") != lineage["lineage_sha256"]:
        raise ValueError("frozen LoRA diagnostic lineage hash mismatch")
    update = value.get("update_count")
    if isinstance(update, bool) or not isinstance(update, int) or update <= 0:
        raise ValueError("frozen LoRA diagnostic requires a trained update_count")
    source = value.get("source_resume")
    if not isinstance(source, Mapping) or set(source) != {"filename", "sha256"}:
        raise ValueError("frozen LoRA diagnostic source resume is malformed")
    if (
        not isinstance(source["filename"], str)
        or re.fullmatch(r"frozen_lora_model_[0-9]+\.pt", source["filename"]) is None
        or not _is_sha256(source["sha256"])
        or source["filename"] != f"frozen_lora_model_{update}.pt"
    ):
        raise ValueError("frozen LoRA diagnostic source resume identity is invalid")
    metadata = value.get("g1_23dof_metadata")
    expected_metadata = {
        "reference_profile": contract.get("reference_profile"),
        "history_length": DEPLOYMENT_HISTORY_LENGTH,
        "observation_layout": OBS_LAYOUT_PADDED_IL29,
        "decoder_input_dim": DEPLOYMENT_DECODER_INPUT_DIM,
        "decoder_output_dim": TARGET_DOF,
        "checkpoint_stage": "diagnostic_adapter_materialization",
        "deployment_ready": False,
        "hardware_authorized": False,
    }
    if not isinstance(metadata, Mapping) or dict(metadata) != expected_metadata:
        raise ValueError("frozen LoRA diagnostic true23 metadata mismatch")
    policy = _copy_policy(value.get("policy_state_dict"))
    policy_hash = inspect_true23_policy_state(
        {"policy_state_dict": policy},
        reference_profile=str(contract.get("reference_profile")),
    )
    if value.get("policy_state_sha256") != policy_hash:
        raise ValueError("frozen LoRA diagnostic policy hash mismatch")
    if policy_hash == contract.get("initial_true23_policy_sha256"):
        raise ValueError("frozen LoRA diagnostic policy is unchanged initialization")
    adapter_hash = value.get("adapter_state_sha256")
    if not _is_sha256(adapter_hash):
        raise ValueError("frozen LoRA diagnostic adapter hash is invalid")
    return value


def load_frozen_lora_diagnostic_policy(path: str | Path) -> Mapping[str, Any]:
    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise ValueError("frozen LoRA diagnostic path may not be a symlink")
    requested = requested.resolve()
    if not requested.is_file():
        raise FileNotFoundError(f"frozen LoRA diagnostic missing: {requested}")
    loaded = torch.load(requested, map_location="cpu", weights_only=True)
    return validate_frozen_lora_diagnostic_policy(loaded)


def materialize_frozen_lora_diagnostic_policy(
    *,
    resume_checkpoint_path: str | Path,
    warm_start_path: str | Path,
    source_checkpoint_path: str | Path,
    output_path: str | Path,
) -> Path:
    resume_path = Path(resume_checkpoint_path).expanduser().resolve()
    if not resume_path.is_file():
        raise FileNotFoundError(f"frozen LoRA resume missing: {resume_path}")
    peek = torch.load(resume_path, map_location="cpu", weights_only=True)
    if not isinstance(peek, Mapping) or CHECKPOINT_HEADER not in peek:
        raise ValueError("source is not a frozen LoRA training checkpoint")
    raw_contract = peek.get("adapter_contract")
    if not isinstance(raw_contract, Mapping):
        raise ValueError("frozen LoRA resume lacks adapter contract")
    rank = raw_contract.get("lora_rank")
    alpha = raw_contract.get("lora_alpha")
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or rank <= 0
        or isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or float(alpha) <= 0.0
    ):
        raise ValueError("frozen LoRA resume adapter dimensions are invalid")
    core = FrozenPlatformTrue23Core(
        warm_start_path=warm_start_path,
        source_checkpoint_path=source_checkpoint_path,
        lora_rank=rank,
        lora_alpha=float(alpha),
    )
    contract = core.adapter_contract()
    checkpoint = load_frozen_platform_lora_checkpoint(
        resume_path,
        expected_contract=contract,
    )
    core.load_lora_state_dict(checkpoint["adapter_state_dict"], strict=True)
    core.assert_frozen_platform_unchanged()
    policy = core.export_true23_policy_state(core.initial_std)
    policy_hash = inspect_true23_policy_state(
        {"policy_state_dict": policy},
        reference_profile=core.reference_profile,
    )
    if policy_hash != checkpoint["merged_true23_policy_sha256"]:
        raise ValueError("materialized policy differs from resume merged hash")
    lineage = validate_mjlab_training_lineage(checkpoint["lineage"])
    artifact = {
        DIAGNOSTIC_HEADER: _header(),
        "policy_state_dict": _copy_policy(policy),
        "policy_state_sha256": policy_hash,
        "g1_23dof_metadata": {
            "reference_profile": core.reference_profile,
            "history_length": DEPLOYMENT_HISTORY_LENGTH,
            "observation_layout": OBS_LAYOUT_PADDED_IL29,
            "decoder_input_dim": DEPLOYMENT_DECODER_INPUT_DIM,
            "decoder_output_dim": TARGET_DOF,
            "checkpoint_stage": "diagnostic_adapter_materialization",
            "deployment_ready": False,
            "hardware_authorized": False,
        },
        "adapter_contract": copy.deepcopy(contract),
        "adapter_contract_sha256": _sha256_mapping(contract),
        "adapter_state_sha256": checkpoint["adapter_state_sha256"],
        "update_count": checkpoint["update_count"],
        "training_lineage": copy.deepcopy(lineage),
        "training_lineage_sha256": lineage["lineage_sha256"],
        "source_resume": {
            "filename": resume_path.name,
            "sha256": sha256_file(resume_path),
        },
    }
    validate_frozen_lora_diagnostic_policy(artifact)
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic policy: {output}")
    if output.is_symlink():
        raise ValueError("diagnostic output path may not be a symlink")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        torch.save(artifact, temporary)
        loaded = torch.load(temporary, map_location="cpu", weights_only=True)
        validate_frozen_lora_diagnostic_policy(loaded)
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        if output.exists():
            raise FileExistsError(f"refusing to overwrite diagnostic policy: {output}")
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return output


def export_frozen_lora_diagnostic_decoder_onnx(
    *,
    diagnostic_policy_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
) -> tuple[Path, Path, Mapping[str, Any]]:
    """Export one merged adapter decoder for simulator comparison only."""
    return _export_frozen_lora_diagnostic_component(
        diagnostic_policy_path=diagnostic_policy_path,
        output_path=output_path,
        report_path=report_path,
        component="decoder",
    )


def export_frozen_lora_diagnostic_encoder_onnx(
    *,
    diagnostic_policy_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
) -> tuple[Path, Path, Mapping[str, Any]]:
    """Export the matching frozen encoder+FSQ; never borrow an unrelated encoder."""
    return _export_frozen_lora_diagnostic_component(
        diagnostic_policy_path=diagnostic_policy_path,
        output_path=output_path,
        report_path=report_path,
        component="encoder",
    )


def _export_frozen_lora_diagnostic_component(
    *,
    diagnostic_policy_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    component: str,
) -> tuple[Path, Path, Mapping[str, Any]]:

    import onnx
    from onnx import shape_inference

    from gear_sonic.utils import g1_23dof_artifact as artifact
    from gear_sonic.utils.g1_true23_diagnostic_pair import ENCODER_CONTRACT

    diagnostic_path = Path(diagnostic_policy_path).expanduser().resolve()
    value = load_frozen_lora_diagnostic_policy(diagnostic_path)
    encoder, decoder, reconstructed_hash = artifact.build_true23_policy_pair(value)
    if reconstructed_hash != value["policy_state_sha256"]:
        raise ValueError("diagnostic component reconstruction hash mismatch")
    if component == "encoder":
        module, input_dim, output_dim = encoder, 267, 64
        input_name, output_name = artifact.ENCODER_ONNX_INPUT_NAME, artifact.ENCODER_ONNX_OUTPUT_NAME
        structure_validator = artifact.validate_encoder_onnx_structure
    elif component == "decoder":
        module, input_dim, output_dim = decoder, DEPLOYMENT_DECODER_INPUT_DIM, TARGET_DOF
        input_name, output_name = artifact.ONNX_INPUT_NAME, artifact.ONNX_OUTPUT_NAME
        structure_validator = artifact.validate_onnx_structure
    else:
        raise ValueError("unknown diagnostic component")
    from gear_sonic.trl.mjlab.frozen_platform_lora_runner import _state_sha256

    required_encoder_hash = _state_sha256(
        {
            name: tensor
            for name, tensor in value["policy_state_dict"].items()
            if name.startswith("actor_module.encoders.teleop.module.")
        }
    )

    output = Path(output_path).expanduser()
    report = Path(report_path).expanduser()
    for path, suffix, context in (
        (output, ".onnx", f"{component} ONNX"),
        (report, ".json", f"{component} report"),
    ):
        if path.is_symlink():
            raise ValueError(f"diagnostic {context} path may not be a symlink")
        if path.suffix.lower() != suffix:
            raise ValueError(f"diagnostic {context} must end in {suffix}")
        lowered = path.name.lower()
        if "diagnostic" not in lowered:
            raise ValueError(f"diagnostic {context} filename must say diagnostic")
        if "promotion" in lowered or "deployment" in lowered:
            raise ValueError(f"diagnostic {context} filename may not imply promotion/deployment")
        if path.exists():
            raise FileExistsError(f"refusing to overwrite diagnostic output: {path}")
    output = output.resolve()
    report = report.resolve()
    if output == report:
        raise ValueError("diagnostic decoder and report paths must differ")
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)

    temporary_onnx: Path | None = None
    temporary_report: Path | None = None
    published: list[Path] = []
    try:
        descriptor, name = tempfile.mkstemp(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp.onnx",
        )
        os.close(descriptor)
        temporary_onnx = Path(name)
        example = torch.zeros(
            1,
            input_dim,
            dtype=torch.float32,
        )
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            torch.onnx.export(
                module,
                example,
                temporary_onnx,
                export_params=True,
                input_names=[input_name],
                output_names=[output_name],
                opset_version=artifact.ONNX_OPSET_VERSION,
                do_constant_folding=True,
                dynamic_axes=None,
                dynamo=False,
            )
        model = onnx.load(temporary_onnx, load_external_data=False)
        model = shape_inference.infer_shapes(
            model,
            strict_mode=True,
            data_prop=True,
        )
        onnx.save_model(model, temporary_onnx, save_as_external_data=False)
        model = onnx.load(temporary_onnx, load_external_data=False)
        structure_validator(model)
        parity = artifact.validate_ort_parity(
            module,
            temporary_onnx,
            input_name=input_name,
            input_dim=input_dim,
            output_name=output_name,
            output_dim=output_dim,
        )
        if component == "encoder" and parity["parity_max_abs_error"] != 0:
            raise ValueError("diagnostic encoder requires exact discrete token parity")
        report_value: dict[str, Any] = {
            "kind": f"g1_true23_frozen_lora_diagnostic_{component}_onnx",
            "schema_version": 1,
            "paired_encoder_state_sha256": required_encoder_hash,
            "encoder_contract": copy.deepcopy(ENCODER_CONTRACT),
            "source": {
                "filename": diagnostic_path.name,
                "sha256": artifact.sha256_file(diagnostic_path),
                "policy_state_sha256": value["policy_state_sha256"],
                "adapter_state_sha256": value["adapter_state_sha256"],
                "update_count": value["update_count"],
            },
            component: {
                "filename": output.name,
                "sha256": artifact.sha256_file(temporary_onnx),
                "input_name": input_name,
                "input_shape": [1, input_dim],
                "output_name": output_name,
                "output_shape": [1, output_dim],
                "opset": artifact.ONNX_OPSET_VERSION,
            },
            "validation": {
                "weights_only_diagnostic_policy_validated": True,
                f"exact_merged_{component}_reconstructed": True,
                "onnx_structure_validated": True,
                "onnx_runtime_parity": parity,
            },
            "diagnostic_only": True,
            "deployment_ready": False,
            "promotion_eligible": False,
            "hardware_authorized": False,
            "active_motor_control_authorized": False,
        }
        descriptor, name = tempfile.mkstemp(
            dir=report.parent,
            prefix=f".{report.name}.",
            suffix=".tmp.json",
        )
        os.close(descriptor)
        temporary_report = Path(name)
        temporary_report.write_bytes(artifact.canonical_json_bytes(report_value))
        for temporary, final in (
            (temporary_onnx, output),
            (temporary_report, report),
        ):
            try:
                os.link(temporary, final)
            except FileExistsError as exc:
                raise FileExistsError(f"refusing to overwrite diagnostic output: {final}") from exc
            temporary.unlink()
            published.append(final)
        temporary_onnx = None
        temporary_report = None
        return output, report, report_value
    except Exception:
        for path in published:
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in (temporary_onnx, temporary_report):
            if path is not None:
                path.unlink(missing_ok=True)


__all__ = [
    "DIAGNOSTIC_HEADER",
    "export_frozen_lora_diagnostic_decoder_onnx",
    "export_frozen_lora_diagnostic_encoder_onnx",
    "load_frozen_lora_diagnostic_policy",
    "materialize_frozen_lora_diagnostic_policy",
    "validate_frozen_lora_diagnostic_policy",
]
