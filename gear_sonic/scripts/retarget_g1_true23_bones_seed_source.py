"""Apply one bounded native23 fit to one hash-bound BONES-SEED source.

Entire source motion, one fixed 2x duration candidate, no excursion reduction.
Failed fits retain diagnostic evidence and never publish an accepted motion.
Source data and derivatives must remain local under the dataset license.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gear_sonic.utils import g1_23dof_task_space_retarget as ik
from gear_sonic.utils.g1_true23_bones_seed import write_json
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_retarget import (
    AdaptationLimits,
    adapt_offline_motion,
    refine_retained_protected_motion,
)
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--refine-retained-directory", type=Path)
    parser.add_argument("--feasibility-restoration", action="store_true")
    args = parser.parse_args(argv)
    output = args.output_directory.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")
    directory = args.source_directory.resolve(strict=True)
    receipt_path, source_path = directory / "report.json", directory / "source.named29.npz"
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("kind") != "g1_true23_bones_seed_named29_source_v1"
        or receipt["output"]["sha256"] != sha256_file(source_path)
        or receipt["indexed_source"]["split"] != "train"
    ):
        raise ValueError("retarget probe requires a hash-bound train-split named29 source")
    bindings = {
        str(path.resolve()): sha256_file(path)
        for path in (
            receipt_path,
            source_path,
            args.source_model,
            args.target_model,
            Path(__file__),
            Path(ik.__file__),
            Path(__file__).resolve().parents[1] / "utils/g1_true23_generalist_retarget.py",
        )
    }
    from gear_sonic.utils.g1_true23_generalist_source_closure import collect_local_source_closure

    root = Path(__file__).resolve().parents[2]
    closure = collect_local_source_closure(root, [Path(__file__).resolve()]).as_source_files(root)
    for value in closure.values():
        path = Path(value)
        bindings[str(path.resolve())] = sha256_file(path)
    source, target = ik.load_models(args.source_model, args.target_model)
    if receipt["source_model"]["sha256"] != sha256_file(args.source_model):
        raise ValueError("retarget source model differs from named source conversion")
    with np.load(source_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    output.mkdir(parents=True, exist_ok=False)
    limits = AdaptationLimits(duration_scales=(2.0,), excursion_scales=(1.0,), maximum_output_frames=2500)
    if args.refine_retained_directory is not None:
        retained = args.refine_retained_directory.resolve(strict=True)
        parent_report_path, parent_arrays_path = retained / "report.json", retained / "fit.diagnostic.npz"
        bindings.update({str(path): sha256_file(path) for path in (parent_report_path, parent_arrays_path)})
        parent_report = json.loads(parent_report_path.read_text())
        if parent_report.get("bones_seed_source_receipt_sha256") != sha256_file(receipt_path):
            raise ValueError("retained fit belongs to another source conversion")
        for path in (source_path, args.source_model, args.target_model):
            if parent_report["input_bindings"].get(str(path.resolve())) != sha256_file(path):
                raise ValueError("retained fit changed source motion or model")
        with np.load(parent_arrays_path, allow_pickle=False) as archive:
            stored = {key: archive[key].copy() for key in archive.files}
        result = refine_retained_protected_motion(
            source_model=source,
            target_model=target,
            arrays=arrays,
            stored=stored,
            forensic_report=parent_report,
            feasibility_restoration=args.feasibility_restoration,
        )
    else:
        result = adapt_offline_motion(
            source_model=source,
            target_model=target,
            arrays=arrays,
            limits=limits,
            source_role="requested_choreography",
            root_reference_refinement=True,
        )
        write_json(output / "initial_fit.json", result.report)
    if args.refine_retained_directory is None and not result.accepted and result.diagnostic_arrays is not None:
        try:
            result = refine_retained_protected_motion(
                source_model=source,
                target_model=target,
                arrays=arrays,
                stored=result.diagnostic_arrays,
                forensic_report=result.report,
                feasibility_restoration=args.feasibility_restoration,
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            result.report["hard_refinement_error"] = f"{type(exc).__name__}: {exc}"
    for path, expected in bindings.items():
        if sha256_file(Path(path)) != expected:
            raise ValueError(f"retarget material changed during fit: {path}")
    report = {
        **result.report,
        "bones_seed_source_receipt_sha256": sha256_file(receipt_path),
        "source_recording_id": receipt["indexed_source"]["recording_id"],
        "source_split": "train",
        "split_sha256": receipt["split_sha256"],
        "input_bindings": bindings,
        "compiled_model_sha256": {
            "source": compiled_model_sha256(source),
            "target": compiled_model_sha256(target),
        },
        "hardware_authorized": False,
        "deployment_ready": False,
    }
    if result.diagnostic_arrays is not None:
        diagnostic_path = output / "fit.diagnostic.npz"
        with diagnostic_path.open("xb") as stream:
            np.savez_compressed(stream, **result.diagnostic_arrays)
        report["diagnostic_output"] = {"path": diagnostic_path.name, "sha256": sha256_file(diagnostic_path)}
    if result.accepted:
        destination = output / "adapted.true23.npz"
        with destination.open("xb") as stream:
            np.savez_compressed(stream, **result.arrays)
        report["output"] = {"path": destination.name, "sha256": sha256_file(destination)}
    write_json(output / "report.json", report)
    print(json.dumps({"path": str(output / "report.json"), "accepted": result.accepted}), flush=True)
    return 0 if result.accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
