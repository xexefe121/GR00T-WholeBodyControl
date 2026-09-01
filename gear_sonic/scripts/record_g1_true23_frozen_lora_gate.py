"""Append one independent checkpoint evaluation to frozen-LoRA gate ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from gear_sonic.utils.g1_true23_frozen_lora_gates import (
    append_frozen_lora_gate_record,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--record", type=Path)
    source.add_argument("--suite-summary", type=Path)
    parser.add_argument("--phase", choices=("breadth", "polish"))
    parser.add_argument("--ledger", type=Path, required=True)
    return parser


def _record_from_suite(path: Path, phase: str | None) -> dict:
    if phase is None:
        raise ValueError("--suite-summary requires --phase")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("kind") != "g1_true23_frozen_lora_comparison_result_v1":
        raise ValueError("suite summary kind mismatch")
    update = value.get("checkpoint_update_count")
    if isinstance(update, bool) or not isinstance(update, int) or update < 0:
        raise ValueError("suite summary update count is invalid")

    def benchmark(name: str) -> dict:
        raw = value.get(name)
        if not isinstance(raw, dict):
            raise ValueError(f"suite summary lacks {name}")
        return {
            "success_rate": raw["success_rate"],
            "mean_tracking_error": raw["mean_tracking_error"],
        }

    second = value.get("second_referee")
    if not isinstance(second, dict):
        raise ValueError("suite summary lacks second referee")
    return {
        "checkpoint": f"{phase}/frozen_lora_model_{update}.pt",
        "update_count": update,
        "phase": phase,
        "in_distribution": benchmark("in_distribution"),
        "tail": benchmark("tail"),
        "out_of_distribution": benchmark("out_of_distribution"),
        "second_referee": {"survival_rate": second["survival_rate"]},
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    ledger_path = args.ledger.expanduser().resolve()
    if args.record is not None:
        if args.phase is not None:
            raise ValueError("--phase is only used with --suite-summary")
        record_path = args.record.expanduser().resolve()
        record = json.loads(record_path.read_text(encoding="utf-8"))
    else:
        record = _record_from_suite(
            args.suite_summary.expanduser().resolve(),
            args.phase,
        )
    existing = (
        json.loads(ledger_path.read_text(encoding="utf-8"))
        if ledger_path.exists()
        else None
    )
    updated = append_frozen_lora_gate_record(existing, record)
    encoded = json.dumps(updated, indent=2, sort_keys=True) + "\n"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = ledger_path.with_suffix(ledger_path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(ledger_path)
    print(encoded, end="")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
