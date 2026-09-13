"""Data ownership and bounded telemetry for the new SIM-only launcher."""

import json
from pathlib import Path

import pytest

from gear_sonic.scripts.train_g1_true23_existing_pico import RECORDINGS, input_capture_selected, validate_bank


@pytest.mark.parametrize("total", (16, 8000))
def test_input_capture_keeps_exact_ends_and_bounded_periodic_samples(total):
    selected = [i for i in range(total) if input_capture_selected(i, total)]
    assert selected[:16] == list(range(16))
    assert selected[-16:] == list(range(total - 16, total))
    assert all(i in selected for i in range(0, total, 64))
    assert len(selected) <= 32 + (total + 63) // 64
    assert all(i < 16 or i >= total - 16 or i % 64 == 0 for i in selected)


def test_actual_bank_preserves_existing_sources_and_excludes008():
    root = Path(__file__).resolve().parents[2]
    bank = validate_bank(root / "artifacts/g1_true23_pico_training_20260910_v1/bank_v1/report.json")
    assert bank["training_recording_ids"] == list(RECORDINGS)
    assert bank["evaluation_data_is_previously_seen_development"]
    assert bank["sensor_only_tracking_claimed"] is False


@pytest.mark.parametrize(
    "key,value",
    (
        ("pico_used_for_training", False),
        ("source_frames", 7265),
        ("all_original29_tasks_bit_exact", False),
        ("training_recording_ids", list(RECORDINGS) + ["existing_pico_derived_walk008"]),
    ),
)
def test_changed_ownership_or_source_coverage_is_rejected(tmp_path, key, value):
    root = Path(__file__).resolve().parents[2]
    original = root / "artifacts/g1_true23_pico_training_20260910_v1/bank_v1/report.json"
    bank = json.loads(original.read_text())
    bank[key] = value
    changed = tmp_path / "bank.json"
    changed.write_text(json.dumps(bank))
    with pytest.raises(ValueError, match="ownership or preservation"):
        validate_bank(changed)
