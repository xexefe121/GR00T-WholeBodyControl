"""Audit selected true23 live-PICO software, fallback drills, and headset gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from gear_sonic.utils.g1_true23_frozen_lora_live_teleop import (
    audit_frozen_lora_live_readiness,
    load_frozen_lora_live_profile,
)


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder-report", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--nominal-report", type=Path, required=True)
    parser.add_argument("--timeout-report", type=Path, required=True)
    parser.add_argument("--gap-report", type=Path, required=True)
    parser.add_argument("--stale-report", type=Path, required=True)
    parser.add_argument("--original-report", type=Path, required=True)
    parser.add_argument("--health-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite readiness report: {output}")
    profile = load_frozen_lora_live_profile(
        decoder_report_path=args.decoder_report,
        candidate_summary_path=args.candidate_summary,
    )
    report = audit_frozen_lora_live_readiness(
        profile=profile,
        nominal_report_path=args.nominal_report,
        fault_report_paths={
            "timeout": args.timeout_report,
            "gap": args.gap_report,
            "stale": args.stale_report,
        },
        original_report_path=args.original_report,
        health_report_path=args.health_report,
    )
    _exclusive_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))  # noqa: T201
    return 0 if report["software_live_teleop_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
