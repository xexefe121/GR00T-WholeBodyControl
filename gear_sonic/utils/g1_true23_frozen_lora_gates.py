"""Independent checkpoint gates for frozen-platform true23 transfer.

Training reward is deliberately absent from checkpoint selection.  A record
contains disjoint in-distribution, tail, OOD, and second-physics referee
scores.  The peak OOD checkpoint wins; two consecutive OOD declines stop the
run even when the in-distribution score remains flat or improves.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FrozenLoraGateConfig:
    breadth_plateau_window: int = 3
    breadth_plateau_tolerance: float = 0.002
    polish_behavior_bank_share: float = 0.10
    in_distribution_success_gate: float = 0.80
    tail_success_gate: float = 0.65
    ood_success_gate: float = 0.60
    second_referee_survival_gate: float = 0.80
    ood_decline_patience: int = 2

    def __post_init__(self) -> None:
        if self.breadth_plateau_window < 2:
            raise ValueError("breadth plateau window must be at least two")
        if self.breadth_plateau_tolerance < 0.0:
            raise ValueError("breadth plateau tolerance must be non-negative")
        for name in (
            "polish_behavior_bank_share",
            "in_distribution_success_gate",
            "tail_success_gate",
            "ood_success_gate",
            "second_referee_survival_gate",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if self.ood_decline_patience != 2:
            raise ValueError("transfer recipe requires exactly two OOD declines")


def frozen_lora_sampling_contract(
    config: FrozenLoraGateConfig | None = None,
) -> dict[str, Any]:
    cfg = config or FrozenLoraGateConfig()
    return {
        "schema_version": 1,
        "breadth": {
            "corpus": "full_retargeted_corpus",
            "sampler": "adaptive_motion_sampler",
            "transition": "in_distribution_plateau",
            "plateau_window": cfg.breadth_plateau_window,
            "plateau_tolerance": cfg.breadth_plateau_tolerance,
        },
        "polish": {
            "corpus": "full_corpus_plus_curated_target_behavior_bank",
            "sampler": "near_uniform_with_fixed_behavior_bank_share",
            "behavior_bank_share": cfg.polish_behavior_bank_share,
            "feasibility_screen_required": True,
            "bank_override_and_dedup_required": True,
            "behavior_bank_takeover_forbidden": True,
        },
        "selection_metric": "out_of_distribution.success_rate",
        "training_reward_used_for_selection": False,
        "stop_after_consecutive_ood_declines": cfg.ood_decline_patience,
        "independent_second_physics_referee_required": True,
        "deployment_ready": False,
        "hardware_authorized": False,
    }


def _rate(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0,1]")
    return result


def _error(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if result < 0.0 or result == float("inf") or result != result:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _benchmark(value: Any, name: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != {
        "success_rate",
        "mean_tracking_error",
    }:
        raise ValueError(
            f"{name} must contain exactly success_rate and mean_tracking_error"
        )
    return {
        "success_rate": _rate(value["success_rate"], f"{name}.success_rate"),
        "mean_tracking_error": _error(
            value["mean_tracking_error"],
            f"{name}.mean_tracking_error",
        ),
    }


def validate_frozen_lora_gate_record(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("gate record must be a mapping")
    expected = {
        "checkpoint",
        "update_count",
        "phase",
        "in_distribution",
        "tail",
        "out_of_distribution",
        "second_referee",
    }
    if set(value) != expected:
        raise ValueError(
            "gate record key mismatch: "
            f"missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )
    checkpoint = value["checkpoint"]
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise ValueError("checkpoint must be a non-empty string")
    update_count = value["update_count"]
    if (
        isinstance(update_count, bool)
        or not isinstance(update_count, int)
        or update_count < 0
    ):
        raise ValueError("update_count must be a non-negative integer")
    phase = value["phase"]
    if phase not in {"breadth", "polish"}:
        raise ValueError("phase must be breadth or polish")
    if Path(checkpoint).name != f"frozen_lora_model_{update_count}.pt":
        raise ValueError("checkpoint filename and update_count differ")
    second = value["second_referee"]
    if not isinstance(second, Mapping) or set(second) != {"survival_rate"}:
        raise ValueError("second_referee must contain exactly survival_rate")
    return {
        "checkpoint": checkpoint,
        "update_count": update_count,
        "phase": phase,
        "in_distribution": _benchmark(
            value["in_distribution"], "in_distribution"
        ),
        "tail": _benchmark(value["tail"], "tail"),
        "out_of_distribution": _benchmark(
            value["out_of_distribution"], "out_of_distribution"
        ),
        "second_referee": {
            "survival_rate": _rate(
                second["survival_rate"], "second_referee.survival_rate"
            )
        },
    }


def _selection_key(record: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        record["out_of_distribution"]["success_rate"],
        record["second_referee"]["survival_rate"],
        record["tail"]["success_rate"],
        record["in_distribution"]["success_rate"],
        -record["out_of_distribution"]["mean_tracking_error"],
        -record["update_count"],
    )


def _plateaued(
    records: Sequence[Mapping[str, Any]],
    config: FrozenLoraGateConfig,
) -> bool:
    breadth = [record for record in records if record["phase"] == "breadth"]
    window = breadth[-config.breadth_plateau_window :]
    if len(window) != config.breadth_plateau_window:
        return False
    values = [record["in_distribution"]["success_rate"] for record in window]
    return max(values) - min(values) <= config.breadth_plateau_tolerance


def build_frozen_lora_gate_ledger(
    records: Sequence[Mapping[str, Any]],
    *,
    config: FrozenLoraGateConfig | None = None,
) -> dict[str, Any]:
    cfg = config or FrozenLoraGateConfig()
    checked = [validate_frozen_lora_gate_record(record) for record in records]
    last_update_by_phase: dict[str, int] = {}
    polish_started = False
    for current in checked:
        phase = current["phase"]
        if phase == "polish":
            polish_started = True
        elif polish_started:
            raise ValueError("breadth gate record may not follow polish")
        previous_update = last_update_by_phase.get(phase)
        if previous_update is not None and current["update_count"] <= previous_update:
            raise ValueError(
                "gate records must have strictly increasing updates within phase"
            )
        last_update_by_phase[phase] = current["update_count"]
    checkpoints = [record["checkpoint"] for record in checked]
    if len(checkpoints) != len(set(checkpoints)):
        raise ValueError("gate record checkpoints must be unique")
    decline_streak = 0
    for previous, current in zip(checked, checked[1:], strict=False):
        if (
            current["out_of_distribution"]["success_rate"]
            < previous["out_of_distribution"]["success_rate"]
        ):
            decline_streak += 1
        else:
            decline_streak = 0
    best = max(checked, key=_selection_key) if checked else None
    training_gate = False
    second_gate = False
    if best is not None:
        training_gate = (
            best["in_distribution"]["success_rate"]
            >= cfg.in_distribution_success_gate
            and best["tail"]["success_rate"] >= cfg.tail_success_gate
            and best["out_of_distribution"]["success_rate"]
            >= cfg.ood_success_gate
        )
        second_gate = (
            best["second_referee"]["survival_rate"]
            >= cfg.second_referee_survival_gate
        )
    return {
        "schema_version": 1,
        "kind": "g1_true23_frozen_lora_independent_gate_ledger",
        "config": asdict(cfg),
        "sampling_contract": frozen_lora_sampling_contract(cfg),
        "records": copy.deepcopy(checked),
        "breadth_plateaued": _plateaued(checked, cfg),
        "recommended_phase": "polish" if _plateaued(checked, cfg) else "breadth",
        "consecutive_ood_declines": decline_streak,
        "stop_training": decline_streak >= cfg.ood_decline_patience,
        "selected_checkpoint": best["checkpoint"] if best is not None else None,
        "training_simulator_gate_passed": training_gate,
        "second_physics_referee_gate_passed": second_gate,
        "promotion_evaluation_eligible": training_gate and second_gate,
        "deployment_ready": False,
        "hardware_authorized": False,
    }


def append_frozen_lora_gate_record(
    ledger: Mapping[str, Any] | None,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    if ledger is None:
        records: list[Mapping[str, Any]] = []
        config = FrozenLoraGateConfig()
    else:
        if ledger.get("kind") != "g1_true23_frozen_lora_independent_gate_ledger":
            raise ValueError("existing gate ledger kind mismatch")
        raw_records = ledger.get("records")
        raw_config = ledger.get("config")
        if not isinstance(raw_records, list) or not isinstance(raw_config, Mapping):
            raise ValueError("existing gate ledger is malformed")
        records = list(raw_records)
        config = FrozenLoraGateConfig(**dict(raw_config))
    records.append(record)
    return build_frozen_lora_gate_ledger(records, config=config)


__all__ = [
    "FrozenLoraGateConfig",
    "append_frozen_lora_gate_record",
    "build_frozen_lora_gate_ledger",
    "frozen_lora_sampling_contract",
    "validate_frozen_lora_gate_record",
]
