"""Use a checkpoint's own v14 encoder/decoder in the common full-request test.

Never label a recovery-trained v14 encoder as the frozen released LoRA
encoder. The established exporter and verifier retain their minimum-update,
exact-policy, source, shape, parity and diagnostic-only gates unchanged.
"""

from dataclasses import replace
import gc
import json
from pathlib import Path

from gear_sonic.envs.mjlab.sonic_true23_causal_history import CAUSAL_HISTORY_PROFILE
from gear_sonic.scripts.train_g1_true23_v14_native_ieee import CONTRACT_KEY, ROOT, comparison_contract
from gear_sonic.utils.g1_23dof_mjlab_diagnostic_onnx import verify_mjlab_diagnostic_onnx
from gear_sonic.utils.g1_23dof_mjlab_training import load_mjlab_training_checkpoint
from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256
from gear_sonic.utils.g1_true23_actuation_profile import SIM_CONFIG, NativeSupportActuationProfile
from gear_sonic.utils.g1_true23_training_precision import EXPECTED_STATE


def require_raw_v14_contract(metadata, resolved, expected_comparison):
    if resolved.get(CONTRACT_KEY) != json.loads(json.dumps(expected_comparison)):
        raise ValueError("v14 evaluation requires the exact original-method comparison contract")
    if resolved.get("stage_one_actuation") != expected_comparison["stage_one_actuation"]:
        # The JSON-roundtrip permits the tuple-to-list change in saved lineage.
        if resolved.get("stage_one_actuation") != json.loads(
            json.dumps(expected_comparison["stage_one_actuation"])
        ):
            raise ValueError("v14 evaluation controller differs from training")
    precision = resolved.get("training_precision", {})
    if (
        precision.get("kind") != "g1_true23_ieee_training_precision_v1"
        or precision.get("requested_state") != EXPECTED_STATE
    ):
        raise ValueError("v14 evaluation requires explicitly IEEE-trained weights")
    if (
        metadata.get("schema_version") != 1
        or "decoder_output_semantics" in metadata.get("contract", {})
        or "safe_target_transform" in metadata.get("contract", {})
    ):
        raise ValueError("v14 common controller requires raw output; an embedded transform would apply twice")
    if metadata["source"]["reference_profile"] != CAUSAL_HISTORY_PROFILE:
        raise ValueError("v14 diagnostic has the wrong causal reference profile")
    for name in ("hardware_authorized", "deployment_ready", "promotion_eligible"):
        if expected_comparison.get(name) is not False:
            raise ValueError("v14 comparison cannot authorize hardware or promotion")


def load_v14_diagnostic_pair(checkpoint_path, metadata_path):
    checkpoint_path = Path(checkpoint_path).resolve(strict=True)
    metadata_path = Path(metadata_path).resolve(strict=True)
    # Verification rejects duplicate JSON keys, unsafe filenames, metadata
    # mutation and graph hashes before these paths may reach the evaluator.
    preliminary = json.loads(metadata_path.read_text())
    encoder = metadata_path.parent / preliminary["artifacts"]["encoder_onnx_filename"]
    decoder = metadata_path.parent / preliminary["artifacts"]["decoder_onnx_filename"]
    metadata = verify_mjlab_diagnostic_onnx(
        encoder,
        decoder,
        metadata_path,
        checkpoint_path=checkpoint_path,
        expected_reference_profile=CAUSAL_HISTORY_PROFILE,
    )
    checkpoint = load_mjlab_training_checkpoint(
        checkpoint_path,
        expected_lineage_sha256=metadata["hashes"]["lineage_sha256"],
        map_location="cpu",
    )
    resolved = checkpoint["lineage"]["materials"]["resolved_config"]["payload"]
    profile = replace(
        NativeSupportActuationProfile.from_sim_config(ROOT / SIM_CONFIG), consistent_controller_state=True
    )
    expected = comparison_contract(resolved[CONTRACT_KEY]["spans"]["path"], profile)
    require_raw_v14_contract(metadata, resolved, expected)
    del checkpoint
    gc.collect()
    return dict(
        kind="g1_true23_original_v14_native_ieee_diagnostic_pair_v1",
        checkpoint=dict(path=str(checkpoint_path), sha256=metadata["hashes"]["checkpoint_sha256"]),
        metadata=dict(path=str(metadata_path), sha256=file_sha256(metadata_path)),
        encoder=dict(path=str(encoder.resolve()), sha256=metadata["hashes"]["encoder_onnx_sha256"]),
        decoder=dict(path=str(decoder.resolve()), sha256=metadata["hashes"]["decoder_onnx_sha256"]),
        checkpoint_update_count=metadata["source"]["checkpoint_update_count"],
        lineage_sha256=metadata["hashes"]["lineage_sha256"],
        policy_state_sha256=metadata["hashes"]["policy_state_sha256"],
        paired_encoder_state_sha256=metadata["hashes"]["encoder_state_sha256"],
        paired_decoder_state_sha256=metadata["hashes"]["decoder_state_sha256"],
        components_from_same_checkpoint=True,
        released_frozen_lora_encoder_substituted=False,
        comparison=expected,
        external_safe_target_transform_required_once=True,
        hardware_authorized=False,
        deployment_ready=False,
        promotion_eligible=False,
    )
