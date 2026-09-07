"""Read-only all-original-timestamp FK audit of one local BONES-SEED fit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gear_sonic.utils import g1_23dof_task_space_retarget as ik
from gear_sonic.utils.g1_true23_bones_seed import write_json
from gear_sonic.utils.g1_true23_bones_seed_source_audit import audit_original_timestamps
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_source_closure import collect_local_source_closure
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-directory", "retarget-directory", "source-model", "target-model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    source_dir, target_dir = (
        args.source_directory.resolve(strict=True),
        args.retarget_directory.resolve(strict=True),
    )
    receipt_path, report_path = source_dir / "report.json", target_dir / "report.json"
    source_path, target_path = source_dir / "source.named29.npz", target_dir / "adapted.true23.npz"
    receipt, report = json.loads(receipt_path.read_text()), json.loads(report_path.read_text())
    bindings = {
        str(path.resolve()): sha256_file(path)
        for path in (
            receipt_path,
            report_path,
            source_path,
            target_path,
            args.source_model,
            args.target_model,
        )
    }
    if (
        receipt.get("kind") != "g1_true23_bones_seed_named29_source_v1"
        or receipt["output"]["sha256"] != sha256_file(source_path)
        or report["output"]["sha256"] != sha256_file(target_path)
        or report["bones_seed_source_receipt_sha256"] != sha256_file(receipt_path)
        or receipt["source_model"]["sha256"] != sha256_file(args.source_model)
    ):
        raise ValueError("source conversion or retarget bindings changed")
    for path in (source_path, args.source_model, args.target_model):
        if report["input_bindings"].get(str(path.resolve())) != sha256_file(path):
            raise ValueError("retarget source/model bytes differ")
    root = Path(__file__).resolve().parents[2]
    closure = collect_local_source_closure(root, [Path(__file__).resolve()]).as_source_files(root)
    bindings.update({str(Path(value).resolve()): sha256_file(Path(value)) for value in closure.values()})
    source_model, target_model = ik.load_models(args.source_model, args.target_model)
    compiled = {"source": compiled_model_sha256(source_model), "target": compiled_model_sha256(target_model)}
    if compiled != report["compiled_model_sha256"]:
        raise ValueError("compiled retarget model changed")
    with np.load(source_path, allow_pickle=False) as archive:
        source = {name: archive[name].copy() for name in archive.files}
    with np.load(target_path, allow_pickle=False) as archive:
        target = {name: archive[name].copy() for name in archive.files}
    result = audit_original_timestamps(source_model, target_model, source, target, report)
    for path, expected in bindings.items():
        if sha256_file(Path(path)) != expected:
            raise ValueError(f"audit material changed: {path}")
    result.update(
        input_bindings=bindings,
        compiled_model_sha256=compiled,
        source_recording_id=receipt["indexed_source"]["recording_id"],
        source_split=receipt["indexed_source"]["split"],
        split_sha256=receipt["split_sha256"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(
        json.dumps(
            {
                "path": str(args.output),
                "passed": result["original_timestamp_fidelity_passed"],
                "original_frames_evaluated": result["original_frames_evaluated"],
                "failures": result["failures"],
            }
        )
    )
    return 0 if result["original_timestamp_fidelity_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
