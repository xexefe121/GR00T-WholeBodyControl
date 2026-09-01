"""Export a merged frozen-LoRA decoder for simulator comparison only."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from gear_sonic.utils.g1_true23_frozen_lora_artifact import (
    export_frozen_lora_diagnostic_decoder_onnx,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output, report, _value = export_frozen_lora_diagnostic_decoder_onnx(
        diagnostic_policy_path=args.diagnostic_policy,
        output_path=args.output,
        report_path=args.report,
    )
    print(f"diagnostic decoder: {output}")  # noqa: T201
    print(f"diagnostic report: {report}")  # noqa: T201
    print("deployment ready: false")  # noqa: T201
    print("active motor control authorized: false")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
