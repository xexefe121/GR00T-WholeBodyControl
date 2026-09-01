"""Create wireless-live gantry sidecar for selected true23 frozen-LoRA policy."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from gear_sonic.scripts.authorize_g1_true23_frozen_lora_dance_gantry import (
    LIVE_KIND,
    active_body_for_operator_mode,
)
from gear_sonic.utils.g1_23dof_artifact import canonical_json_bytes


def active_body(args: argparse.Namespace) -> dict:
    return active_body_for_operator_mode(args, direct_dance=False)


def _object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--promotion", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--decoder-report", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--happy-dance-report", type=Path, required=True)
    parser.add_argument("--happy-dance-trajectory", type=Path, required=True)
    parser.add_argument("--live-qualification", type=Path, required=True)
    parser.add_argument("--packet-bundle", type=Path, required=True)
    parser.add_argument("--live-shadow-evidence", type=Path, required=True)
    parser.add_argument("--authorization-id", required=True)
    parser.add_argument("--gantry-authorize", required=True)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    expected = active_body(args)
    if expected.get("kind") != LIVE_KIND:
        raise ValueError("wireless-live sidecar kind mismatch")
    output = args.output.expanduser().resolve()
    if args.verify_only:
        if _object(output) != expected:
            raise ValueError("live active sidecar differs from re-verified evidence")
        print(output)
        return 0
    if os.path.lexists(output):
        raise FileExistsError("refusing to overwrite live active sidecar")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(canonical_json_bytes(expected))
        stream.flush()
        os.fsync(stream.fileno())
    print(output)
    print("operator contract: wireless L2/A; free-standing authorization: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
