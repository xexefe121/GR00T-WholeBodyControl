"""Fresh SIM-only world-quality training with explicit failure sampling state."""

import copy
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from gear_sonic.envs.mjlab.sonic_true23_failure_sampling import FailureAdaptiveRootFeedbackCommand
from gear_sonic.trl.mjlab.native23_world_quality_runner import (
    CHECKPOINT_HEADER as QUALITY_HEADER,
    Native23WorldQualityRunner,
    validate_checkpoint as validate_quality_checkpoint,
)
from gear_sonic.utils.g1_true23_failure_sampling import FailureBinSampler, sampling_contract
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_training_precision import write_runtime

CHECKPOINT_HEADER = {**QUALITY_HEADER, "kind": "g1_native23_world_quality_failure_sampling_snapshot"}


def require_contract(resolved):
    if resolved.get("native23_failure_sampling") != sampling_contract():
        raise ValueError("failure sampling contract missing or changed")
    if resolved["native23_root_feedback"]["start_schedule"] != sampling_contract():
        raise ValueError("executed reset distribution must not be labeled uniform")


def quality_schema_view(checkpoint):
    if not isinstance(checkpoint, dict) or checkpoint.get("header") != CHECKPOINT_HEADER:
        raise ValueError("failure sampling reader requires its distinct research header")
    require_contract(checkpoint["lineage"]["materials"]["resolved_config"]["payload"])
    if "failure_sampling_state" not in checkpoint:
        raise ValueError("failure sampling snapshot is missing sampler state")
    # Validate the common tensor schema without relabeling any artifact on disk.
    return {
        **{k: v for k, v in checkpoint.items() if k != "failure_sampling_state"},
        "header": copy.deepcopy(QUALITY_HEADER),
    }


def validate_sampler_state(checkpoint):
    resolved = checkpoint["lineage"]["materials"]["resolved_config"]["payload"]
    state = checkpoint["failure_sampling_state"]
    if (
        set(state) != {"contract", "controls", "first", "lengths", "ema", "total_failures"}
        or state["contract"] != sampling_contract()
    ):
        raise ValueError("failure sampling state schema changed")
    spans = resolved["native23_root_feedback"]["curriculum"]["derived_spans"]["spans"]
    first = torch.tensor([r["start"] + r["timeline"]["source_start_frame"] - 1 for r in spans])
    lengths = torch.tensor([r["timeline"]["source_frames"] for r in spans])
    sampler = FailureBinSampler(first, lengths, device="cpu")
    for key, reference in (("first", first), ("lengths", lengths)):
        if (
            not isinstance(state[key], torch.Tensor)
            or state[key].dtype != torch.long
            or not torch.equal(state[key], reference)
        ):
            raise ValueError("failure sampler source boundaries differ from original source")
    controls = state["controls"]
    if (
        type(controls) is not int
        or controls < 0
        or controls != checkpoint["trainer_state"]["env_common_step_counter"]
    ):
        raise ValueError("failure sampler actual-control counter differs")
    for key, dtype in (("ema", torch.float64), ("total_failures", torch.long)):
        value = state[key]
        if not isinstance(value, torch.Tensor) or value.shape != sampler.ema.shape or value.dtype != dtype:
            raise ValueError("failure sampler statistic shape or dtype differs")
        if not torch.isfinite(value).all() or (value < 0).any() or (value[sampler.bin_lengths == 0] != 0).any():
            raise ValueError("failure sampler has nonfinite, negative or padded statistics")
    if (
        state["total_failures"].sum() > controls * resolved["num_envs"]
        or state["ema"].max() > resolved["num_envs"]
        or (state["ema"] > state["total_failures"].double()).any()
    ):
        raise ValueError("failure sampler statistics exceed observed environment controls")
    return state


def validate_checkpoint(checkpoint, *, actor, lineage, minimum_update_count=0):
    require_contract(lineage["materials"]["resolved_config"]["payload"])
    validate_quality_checkpoint(
        quality_schema_view(checkpoint), actor=actor, lineage=lineage, minimum_update_count=minimum_update_count
    )
    validate_sampler_state(checkpoint)
    return checkpoint


class Native23FailureSamplingRunner(Native23WorldQualityRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env.unwrapped.command_manager.get_term("motion").install_failure_monitor()

    def require_runtime(self):
        super().require_runtime()
        require_contract(self.training_lineage["materials"]["resolved_config"]["payload"])
        if not isinstance(
            self.env.unwrapped.command_manager.get_term("motion"), FailureAdaptiveRootFeedbackCommand
        ):
            raise ValueError("declared failure-weighted command is not running")

    def _numbered_checkpoint_path(self, update_count):
        old = super()._numbered_checkpoint_path(update_count)
        return old.with_name(old.name.replace("world_quality_model_", "failure_sampling_model_"))

    def _checkpoint_payload(self):
        result = super()._checkpoint_payload()
        result["header"] = copy.deepcopy(CHECKPOINT_HEADER)
        result["failure_sampling_state"] = self.env.unwrapped.command_manager.get_term(
            "motion"
        ).require_sampling_runtime()
        return result

    def save(self, path, infos=None):
        self.require_runtime()
        if infos is not None:
            raise ValueError("failure sampling snapshots forbid stock RSL infos")
        self._require_checkpointable()
        output = Path(path).expanduser()
        if output.exists() or output.is_symlink():
            raise FileExistsError("failure sampling snapshot refuses overwrite")
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False
            ) as stream:
                temporary = Path(stream.name)
            torch.save(self._checkpoint_payload(), temporary)
            loaded = torch.load(temporary, map_location="cpu", weights_only=True)
            validate_checkpoint(loaded, actor=self.alg.get_policy(), lineage=self.training_lineage)
            with temporary.open("rb+") as stream:
                os.fsync(stream.fileno())
            os.link(temporary, output)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        self._last_checkpoint_path = output
        self._last_checkpoint_update_count = self._require_counter_coherence()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        command = self.env.unwrapped.command_manager.get_term("motion")
        command.require_sampling_runtime()
        try:
            return super().learn(num_learning_iterations, init_at_random_ep_len)
        finally:
            count = self.completed_update_count
            arrays = command.capture_failure_sampling()
            target = self.checkpoint_dir.parent / f"failure_sampling_after_{count}_updates.npz"
            with target.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            write_runtime(
                target.with_suffix(".json"),
                dict(
                    kind="native23_actual_failure_sampling_capture_v1",
                    contract=sampling_contract(),
                    completed_updates=count,
                    actual_controls=command.failure_sampler.controls,
                    actual_source_failures=int(command.failure_sampler.total_failures.sum()),
                    arrays_path=str(target),
                    arrays_sha256=sha256_file(target),
                    lineage_sha256=self.lineage_sha256,
                    training_state_poisoned=bool(self._training_state_poisoned),
                    independent_audit_passed=False,
                    deployment_ready=False,
                    hardware_authorized=False,
                ),
            )
