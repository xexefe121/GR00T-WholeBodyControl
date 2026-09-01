from pathlib import Path

import torch

from gear_sonic.trl.mjlab.frozen_platform_lora_actor import (
    FrozenPlatformTrue23Core,
)
from gear_sonic.trl.mjlab.frozen_platform_lora_runner import (
    CHECKPOINT_HEADER,
    _checkpoint_header,
    _contract_sha256,
    _state_sha256,
)
from gear_sonic.utils.g1_23dof_artifact import (
    build_true23_policy_pair,
    inspect_true23_policy_state,
)
from gear_sonic.utils.g1_23dof_contract import MINIMUM_TRAINING_UPDATES
from gear_sonic.utils.g1_23dof_mjlab_training import (
    build_file_manifest,
    build_mjlab_training_lineage,
)
from gear_sonic.utils.g1_true23_frozen_lora_artifact import (
    export_frozen_lora_diagnostic_decoder_onnx,
    load_frozen_lora_diagnostic_policy,
    materialize_frozen_lora_diagnostic_policy,
)


def test_materializer_reconstructs_hash_bound_diagnostic_policy(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    warm_start = root / "sonic_release/g1_23dof_rev_1_0_init.pt"
    source = root / "sonic_release/last.pt"
    core = FrozenPlatformTrue23Core(
        warm_start_path=warm_start,
        source_checkpoint_path=source,
        lora_rank=16,
        lora_alpha=16.0,
    )
    adapter = core.lora_state_dict()
    first_b = next(name for name in adapter if name.endswith("lora_b"))
    adapter[first_b][0, 0] = 1.0e-4
    core.load_lora_state_dict(adapter, strict=True)
    contract = core.adapter_contract()
    merged = core.export_true23_policy_state(core.initial_std)
    merged_hash = inspect_true23_policy_state(
        {"policy_state_dict": merged},
        reference_profile=core.reference_profile,
    )

    source_file = tmp_path / "runner.py"
    asset_file = tmp_path / "g1.xml"
    dataset_file = tmp_path / "motion.npz"
    source_file.write_text("FROZEN_LORA = True\n", encoding="utf-8")
    asset_file.write_text("<mujoco/>\n", encoding="utf-8")
    dataset_file.write_bytes(b"motion")
    lineage = build_mjlab_training_lineage(
        warm_start,
        resolved_config={
            "task": "Unitree-G1-23Dof-Frozen-LoRA",
            "output_dim": 23,
        },
        source_manifest=build_file_manifest(
            {"runner.py": source_file}, kind="source_files"
        ),
        asset_manifest=build_file_manifest(
            {"g1.xml": asset_file}, kind="robot_assets"
        ),
        dataset_manifest=build_file_manifest(
            {"motion.npz": dataset_file}, kind="motion_dataset"
        ),
    )
    critic = {"value.weight": torch.ones(1, 1)}
    update = MINIMUM_TRAINING_UPDATES
    resume = tmp_path / f"frozen_lora_model_{update}.pt"
    torch.save(
        {
            CHECKPOINT_HEADER: _checkpoint_header(),
            "adapter_contract": contract,
            "adapter_contract_sha256": _contract_sha256(contract),
            "adapter_state_dict": adapter,
            "adapter_state_sha256": _state_sha256(adapter),
            "critic_state_dict": critic,
            "critic_state_sha256": _state_sha256(critic),
            "optimizer_state_dict": {"state": {}, "param_groups": []},
            "update_count": update,
            "trainer_state": {
                "completed_update_count": update,
                "current_learning_iteration": update,
                "env_common_step_counter": 1000,
                "algorithm_learning_rate": 5.0e-6,
            },
            "lineage": lineage,
            "lineage_sha256": lineage["lineage_sha256"],
            "merged_true23_policy_sha256": merged_hash,
        },
        resume,
    )
    output = tmp_path / "candidate.diagnostic.pt"

    materialize_frozen_lora_diagnostic_policy(
        resume_checkpoint_path=resume,
        warm_start_path=warm_start,
        source_checkpoint_path=source,
        output_path=output,
    )
    artifact = load_frozen_lora_diagnostic_policy(output)
    encoder, decoder, reconstructed_hash = build_true23_policy_pair(artifact)

    assert reconstructed_hash == merged_hash
    assert artifact["update_count"] == update
    assert artifact["source_resume"]["filename"] == resume.name
    assert artifact["g1_23dof_metadata"]["deployment_ready"] is False
    assert encoder(torch.zeros(1, 267)).shape == (1, 64)
    assert decoder(torch.zeros(1, 994)).shape == (1, 23)

    decoder_output = tmp_path / "candidate.diagnostic.decoder.onnx"
    decoder_report = tmp_path / "candidate.diagnostic.decoder.json"
    _decoder_path, _report_path, report = (
        export_frozen_lora_diagnostic_decoder_onnx(
            diagnostic_policy_path=output,
            output_path=decoder_output,
            report_path=decoder_report,
        )
    )
    assert decoder_output.is_file()
    assert decoder_report.is_file()
    assert report["source"]["policy_state_sha256"] == merged_hash
    assert report["validation"]["onnx_runtime_parity"][
        "parity_case_count"
    ] == 3
    assert report["deployment_ready"] is False
    assert report["active_motor_control_authorized"] is False
