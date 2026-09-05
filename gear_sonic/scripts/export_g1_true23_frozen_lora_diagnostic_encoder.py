"""Export the encoder+FSQ paired with a frozen-LoRA diagnostic checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from gear_sonic.utils.g1_true23_frozen_lora_artifact import export_frozen_lora_diagnostic_encoder_onnx


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    output, report, _ = export_frozen_lora_diagnostic_encoder_onnx(
        diagnostic_policy_path=args.diagnostic_policy,
        output_path=args.output,
        report_path=args.report,
    )
    print(f"Diagnostic encoder: {output}\nReport: {report}\nHardware authorized: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
