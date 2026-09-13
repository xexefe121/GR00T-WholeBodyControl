"""Compare immutable measured traces; no control, resimulation or promotion."""

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_checkpoint import metrics
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "g1_true23_pico_training_20260910_v1/eval500_v1"
BASE = Path("/mnt/c/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", type=int, choices=(600, 1000), required=True)
    args = parser.parse_args()
    output = HERE / f"comparison_{args.update}.json"
    if output.exists():
        raise FileExistsError("measured comparison refuses overwrite")
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = sha256_file(path)
        if expected is not None and digest != expected:
            raise ValueError("comparison input changed: " + str(path))
        inputs[str(path)] = digest
        return path

    def arrays(path, expected):
        with np.load(bind(path, expected), allow_pickle=False) as bundle:
            return {key: bundle[key].copy() for key in bundle.files}

    def measured(directory, update=None):
        report = json.loads(bind(directory / "report.json").read_text())
        if update is not None:
            assert report["identity"]["completed_training_updates"] == update
        path = directory / "trace.npz"
        digest = report.get("trace_sha256") or report["inputs"][str(path)]
        trace = arrays(path, digest)
        assert len(trace["qpos"]) - 1 == report["result"]["completed_controls"]
        return report, trace

    bind(__file__)
    bind(HERE / "evaluate_checkpoint.py")
    audit = json.loads(bind(HERE / f"eval{args.update}_v1/independent_audit.json").read_text())
    bind(PARENT / "independent_audit.json")
    rows = []
    for name in ("walk002", "walk003", "walk008", "pico"):
        candidate, candidate_trace = measured(HERE / f"eval{args.update}_v1" / name, args.update)
        parent, parent_trace = measured(PARENT / name, 500)
        original_directory = BASE / "released_core_comparison_v1/normal" / name
        if name == "pico":
            original_directory = BASE / "normal_core_pico_v1"
        original, original_trace = measured(original_directory)
        reports = dict(candidate=candidate, parent500=parent, untouched_sonic=original)
        traces = dict(candidate=candidate_trace, parent500=parent_trace, untouched_sonic=original_trace)
        timeline = candidate["timeline"]
        for label, report in reports.items():
            assert report["timeline"] == timeline
            for key in ("qpos", "qvel"):
                np.testing.assert_array_equal(traces[label][key][0], candidate_trace[key][0])
            for key in ("model_sha256", "compiled_model_sha256", "physics_config_sha256", "motion_sha256"):
                assert report["result"][key] == candidate["result"][key]
        motion = arrays(timeline["timeline_path"], timeline["timeline_sha256"])
        phase = next(row for row in timeline["phases"] if row["name"] == "source_motion")
        common = min(len(trace["qpos"]) for trace in traces.values()) - 1
        parent_common = min(len(candidate_trace["qpos"]), len(parent_trace["qpos"])) - 1
        row = dict(
            name=name,
            requested_source_controls=phase["requested_controls"],
            common_three_way_controls=common,
            matched_three_way={key: metrics(trace, motion, phase, common) for key, trace in traces.items()},
            common_parent_candidate_controls=parent_common,
            matched_parent_candidate={
                key: metrics(traces[key], motion, phase, parent_common) for key in ("candidate", "parent500")
            },
            entire_executed_source={
                key: metrics(trace, motion, phase, len(trace["qpos"]) - 1) for key, trace in traces.items()
            },
            completed_controls={key: report["result"]["completed_controls"] for key, report in reports.items()},
            requested_controls=candidate["result"]["requested_controls"],
            candidate_failure=candidate["result"]["failure"],
            candidate_tracking_passed=candidate["lifecycle"]["source_motion_tracking"][
                "provisional_reference_landmark_screen_passed"
            ],
            optimizer_recording=name != "walk008",
            previously_seen_development_evaluation=True,
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = dict(
        kind="foot_precision_preserved_learner_measured_comparison_v1",
        update=args.update,
        cases=rows,
        inputs=inputs,
        replay_audit=audit,
        same_initial_physics_and_reference=True,
        changed_training_objective_and_additional_budget=True,
        causal_reward_ablation_claimed=False,
        prefix_metrics_are_not_full_motion_success=True,
        deployment_ready=False,
        hardware_authorized=False,
        simulator_qualified=False,
    )
    for path, expected in inputs.items():
        assert sha256_file(Path(path)) == expected, path
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
