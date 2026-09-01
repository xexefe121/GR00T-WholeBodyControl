"""Merge one frozen-LoRA resume into simulator-evaluation policy weights."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from gear_sonic.utils.g1_true23_frozen_lora_artifact import (
    materialize_frozen_lora_diagnostic_policy,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = materialize_frozen_lora_diagnostic_policy(
        resume_checkpoint_path=args.checkpoint,
        warm_start_path=args.warm_start,
        source_checkpoint_path=args.source_checkpoint,
        output_path=args.output,
    )
    print(output)  # noqa: T201
    print("deployment ready: false")  # noqa: T201
    print("hardware authorized: false")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
