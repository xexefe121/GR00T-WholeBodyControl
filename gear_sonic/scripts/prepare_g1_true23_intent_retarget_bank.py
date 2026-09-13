"""Prepare an explicit retarget-native / immutable-original29 training bank."""

import argparse
import json
from pathlib import Path

from gear_sonic.utils.g1_true23_intent_retarget_bank import prepare_bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-bank-report", type=Path, required=True)
    parser.add_argument("--retarget-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = prepare_bank(args.previous_bank_report, args.retarget_root, args.output)
    print(json.dumps({key: report[key] for key in ("kind", "source_frames", "lifecycle_frames", "checks", "files")}, indent=2))


if __name__ == "__main__":
    main()
