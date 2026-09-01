import pytest

from gear_sonic.utils.g1_true23_frozen_lora_gates import (
    append_frozen_lora_gate_record,
    build_frozen_lora_gate_ledger,
)


def _record(
    update: int,
    *,
    phase: str,
    in_distribution: float,
    tail: float,
    ood: float,
    referee: float,
) -> dict:
    return {
        "checkpoint": f"frozen_lora_model_{update}.pt",
        "update_count": update,
        "phase": phase,
        "in_distribution": {
            "success_rate": in_distribution,
            "mean_tracking_error": 1.0 - in_distribution,
        },
        "tail": {
            "success_rate": tail,
            "mean_tracking_error": 1.0 - tail,
        },
        "out_of_distribution": {
            "success_rate": ood,
            "mean_tracking_error": 1.0 - ood,
        },
        "second_referee": {"survival_rate": referee},
    }


def test_breadth_plateau_recommends_polish() -> None:
    records = [
        _record(
            update,
            phase="breadth",
            in_distribution=score,
            tail=0.70,
            ood=0.62,
            referee=0.85,
        )
        for update, score in ((100, 0.800), (200, 0.801), (300, 0.800))
    ]
    ledger = build_frozen_lora_gate_ledger(records)

    assert ledger["breadth_plateaued"] is True
    assert ledger["recommended_phase"] == "polish"
    assert ledger["promotion_evaluation_eligible"] is True
    assert ledger["deployment_ready"] is False
    assert ledger["hardware_authorized"] is False


def test_two_consecutive_ood_declines_stop_and_keep_peak_checkpoint() -> None:
    records = [
        _record(
            100,
            phase="polish",
            in_distribution=0.90,
            tail=0.75,
            ood=0.69,
            referee=0.90,
        ),
        _record(
            200,
            phase="polish",
            in_distribution=0.91,
            tail=0.76,
            ood=0.675,
            referee=0.90,
        ),
        _record(
            300,
            phase="polish",
            in_distribution=0.92,
            tail=0.77,
            ood=0.658,
            referee=0.91,
        ),
    ]
    ledger = build_frozen_lora_gate_ledger(records)

    assert ledger["consecutive_ood_declines"] == 2
    assert ledger["stop_training"] is True
    assert ledger["selected_checkpoint"] == "frozen_lora_model_100.pt"


def test_ledger_append_revalidates_existing_contract() -> None:
    first = _record(
        100,
        phase="breadth",
        in_distribution=0.8,
        tail=0.7,
        ood=0.6,
        referee=0.8,
    )
    ledger = append_frozen_lora_gate_record(None, first)
    second = _record(
        200,
        phase="breadth",
        in_distribution=0.81,
        tail=0.71,
        ood=0.61,
        referee=0.81,
    )

    updated = append_frozen_lora_gate_record(ledger, second)
    assert len(updated["records"]) == 2

    with pytest.raises(ValueError, match="strictly increasing"):
        append_frozen_lora_gate_record(updated, second)


def test_polish_has_fresh_phase_local_update_counter() -> None:
    breadth = _record(
        500,
        phase="breadth",
        in_distribution=0.8,
        tail=0.7,
        ood=0.6,
        referee=0.8,
    )
    breadth["checkpoint"] = "breadth/frozen_lora_model_500.pt"
    polish = _record(
        100,
        phase="polish",
        in_distribution=0.81,
        tail=0.71,
        ood=0.61,
        referee=0.81,
    )
    polish["checkpoint"] = "polish/frozen_lora_model_100.pt"

    ledger = build_frozen_lora_gate_ledger([breadth, polish])
    assert len(ledger["records"]) == 2
