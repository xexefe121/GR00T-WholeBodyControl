"""Recompute full original-planner FK after retiming, without simulator stepping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.scripts.retarget_g1_true23_generalist_planned_trace import planned_named_source
from gear_sonic.utils.g1_true23_bones_seed_source_audit import audit_original_timestamps
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_source_closure import collect_local_source_closure
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256


def validate_planned_bindings(report, paths, bindings):
    if (
        report.get("accepted") is not True
        or report.get("source_field") != "planned_qpos50"
        or report.get("recorded_policy_pose_used_as_choreography") is not False
        or report.get("adapted_motion_sha256") != bindings[str(paths["adapted"])]
        or report.get("named_source_sha256") != bindings[str(paths["named"])]
        or any(
            report.get("input_bindings", {}).get(str(paths[key])) != bindings[str(paths[key])]
            for key in ("trace", "source_model", "target_model")
        )
    ):
        raise ValueError("requires an accepted hash-bound original planned29 source, not recorded policy poses")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("retarget-directory", "trace", "source-model", "target-model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    directory = args.retarget_directory.resolve(strict=True)
    paths = {
        "report": directory / "report.json",
        "adapted": directory / "adapted.true23.npz",
        "named": directory / "planned.named29.npz",
        "trace": args.trace.resolve(strict=True),
        "source_model": args.source_model.resolve(strict=True),
        "target_model": args.target_model.resolve(strict=True),
    }
    bindings = {str(path): sha256_file(path) for path in paths.values()}
    closure = collect_local_source_closure(Path(__file__).resolve().parents[2], [Path(__file__).resolve()])
    bindings.update({str(path): sha256_file(path) for path in closure.files})
    report = json.loads(paths["report"].read_text())
    validate_planned_bindings(report, paths, bindings)
    source_model = (
        mujoco.MjModel.from_binary_path(str(paths["source_model"]))
        if paths["source_model"].suffix == ".mjb"
        else mujoco.MjModel.from_xml_path(str(paths["source_model"]))
    )
    target_model = mujoco.MjModel.from_xml_path(str(paths["target_model"]))
    compiled = {"source": compiled_model_sha256(source_model), "target": compiled_model_sha256(target_model)}
    if compiled != report["compiled_models"]:
        raise ValueError("compiled source or native23 retarget model changed")
    with np.load(paths["trace"], allow_pickle=False) as trace:
        reconstructed = planned_named_source(trace, source_model)
    with np.load(paths["named"], allow_pickle=False) as archive:
        named = {key: archive[key].copy() for key in archive.files}
    if set(named) != set(reconstructed) or any(
        not np.array_equal(named[key], reconstructed[key]) for key in reconstructed
    ):
        raise ValueError("named source differs from complete original planned_qpos50")
    with np.load(paths["adapted"], allow_pickle=False) as archive:
        adapted = {key: archive[key].copy() for key in archive.files}
    result = audit_original_timestamps(source_model, target_model, named, adapted, report)
    result.update(
        kind="g1_true23_planned29_all_original_timestamps_fk_audit_v1",
        source_field="planned_qpos50",
        recorded_policy_pose_used_as_choreography=False,
        input_bindings=bindings,
        compiled_models=compiled,
        original_speed_policy_parity_proven=False,
    )
    for path, expected in bindings.items():
        if sha256_file(Path(path)) != expected:
            raise ValueError(f"original-time audit input changed: {path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "passed": result["original_timestamp_fidelity_passed"],
                "original_frames_evaluated": result["original_frames_evaluated"],
                "failures": result["failures"],
            }
        ),
        flush=True,
    )
    return 0 if result["original_timestamp_fidelity_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
