"""Inventory explicit motion banks or audit a provenance-rich generalist manifest.

Examples::

    python -m gear_sonic.scripts.audit_g1_true23_generalist_corpus inventory \
        --root /path/to/released/motions --hash-files --output inventory.json
    python -m gear_sonic.scripts.audit_g1_true23_generalist_corpus audit \
        --manifest /path/to/manifest.json --output audit.json

``audit`` returns 0 for an integrity-valid manifest, 2 for invalid input.
``--require-generalization-corpus`` additionally requires the quantity/coverage
gate; it does not prove training quality or physical readiness. Output is always
created exclusively. No pickle is loaded and no dataset is downloaded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gear_sonic.utils.g1_true23_generalist_corpus import audit_manifest, inventory_roots


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("inventory")
    inventory.add_argument("--root", action="append", type=Path, required=True)
    inventory.add_argument("--maximum-files", type=int, default=10000)
    inventory.add_argument("--hash-files", action="store_true")
    inventory.add_argument("--output", type=Path, required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--require-generalization-corpus", action="store_true")
    audit.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    try:
        if args.command == "inventory":
            report = inventory_roots(args.root, maximum_files=args.maximum_files, hash_files=args.hash_files)
            status = 2 if report["truncated"] else 0
        else:
            manifest_path = args.manifest.resolve(strict=True)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            report = audit_manifest(manifest, manifest_path.parent)
            status = (
                2
                if args.require_generalization_corpus and not report["corpus_quantity_and_coverage_sufficient"]
                else 0
            )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        report = {
            "schema_version": 1,
            "kind": "g1_true23_generalist_corpus_input_rejected",
            "manifest_valid": False,
            "error": f"{type(exc).__name__}: {exc}",
            "simulator_qualification_complete": False,
            "hardware_authorized": False,
        }
        status = 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(args.output.resolve())
    return status


if __name__ == "__main__":
    raise SystemExit(main())
