"""Reproduce all nominal physical steps before any trajectory optimization."""

import argparse
import json
from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_source_closure import collect_local_source_closure
from gear_sonic.utils.g1_true23_native_model_actuation import NativeModelActuationProfile
from gear_sonic.utils.g1_true23_pd_shooting import PdShootingPlant
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-directory", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    source_report = (args.evaluation_directory / "report.json").resolve(strict=True)
    report = json.loads(source_report.read_text())
    row = next(r for r in report["records"] if r["case"] == "nominal")
    result = row["result"]
    if result["failure"] is not None or result["completed_controls"] != result["available_controls"]:
        raise ValueError("PD shooting initialization requires an actually completed full nominal baseline")
    trace = Path(row["trace_path"]).resolve(strict=True)
    if sha256_file(trace) != row["trace_sha256"]:
        raise ValueError("PD shooting baseline trace changed")
    _, model, _ = prepare_true23_model(args.asset_root / MODEL, root / PHYSICS)
    plant = PdShootingPlant(model, NativeModelActuationProfile.from_sim_config(root / PHYSICS))
    if plant.model_sha != result["compiled_model_sha256"]:
        raise ValueError("PD shooting baseline uses different physics")
    paths = [
        source_report,
        trace,
        args.asset_root / MODEL,
        root / PHYSICS,
        *collect_local_source_closure(root, [Path(__file__)]).files,
    ]
    bindings = {str(path.resolve()): sha256_file(path) for path in paths}
    args.output_directory.mkdir(parents=True, exist_ok=False)
    with np.load(trace, allow_pickle=False) as archive:
        original = {key: archive[key].copy() for key in archive.files}
    if np.any(original["physics_external_force_world_n"]):
        raise ValueError("nominal PD shooting cannot import a pushed baseline")
    initial = plant.state(plant.initial_data(original["qpos"][0], original["qvel"][0]))
    arrays = plant.rollout(initial, original["target23"], physics_records=True)
    keys = sorted(set(arrays) & set(original))
    mismatches = [
        key
        for key in keys
        if arrays[key].shape != original[key].shape
        or arrays[key].dtype != original[key].dtype
        or arrays[key].tobytes() != original[key].tobytes()
    ]
    if mismatches:
        difference = {
            key: float(np.max(np.abs(arrays[key].astype(float) - original[key].astype(float))))
            for key in mismatches
            if arrays[key].shape == original[key].shape
        }
        raise ValueError("PD shooting did not reproduce baseline: " + str(difference))
    if len(keys) != 12 or len(arrays["qpos"]) != result["available_controls"] + 1:
        raise ValueError("PD shooting comparison must cover all twelve physical arrays and the entire timeline")
    arrays["target23"] = original["target23"]
    if any(sha256_file(Path(path)) != digest for path, digest in bindings.items()):
        raise ValueError("PD shooting source or implementation changed during verification")
    output_trace = args.output_directory / "reproduction.npz"
    with output_trace.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    evidence = dict(
        kind="g1_true23_pd_shooting_full_baseline_reproduction_v1",
        inputs=bindings,
        parent_evaluation=str(source_report),
        parent_trace=str(trace),
        parent_trace_sha256=row["trace_sha256"],
        original_policy_identity=row["policy_identity"],
        complete_controls=result["available_controls"],
        physical_arrays_bit_exact=keys,
        full_integration_state_including_warmstart_recorded=True,
        reproduced_physics_sha256=plant.model_sha,
        trace_path=str(output_trace.resolve()),
        trace_sha256=sha256_file(output_trace),
        open_loop_recorded_command_replay_not_a_policy=True,
        state_writes_after_initial_condition=0,
        original_tracking_failures_not_reclassified=True,
        reference_or_policy_modified=False,
        simulator_qualified=False,
        hardware_authorized=False,
        deployment_ready=False,
    )
    with (args.output_directory / "report.json").open("x") as stream:
        json.dump(evidence, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {key: evidence[key] for key in ("complete_controls", "physical_arrays_bit_exact", "trace_sha256")}
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
