"""Compare both declared milestones on identical motion prefixes, no new physics."""

import json
from pathlib import Path

import numpy as np

from evaluate_checkpoint import metrics
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file

HERE = Path(__file__).resolve().parent


def main():
    output = HERE / "milestone_comparison.json"
    if output.exists():
        raise FileExistsError("milestone comparison refuses overwrite")
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = sha256_file(path)
        if expected is not None:
            assert digest == expected, path
        inputs[str(path)] = digest
        return path

    def arrays(path, expected):
        with np.load(bind(path, expected), allow_pickle=False) as z:
            return {k: z[k].copy() for k in z.files}

    bind(__file__)
    bind(HERE / "evaluate_checkpoint.py")
    rows = []
    for name in ("walk002", "walk003", "walk008", "pico"):
        reports, traces = [], []
        for update in (100, 500):
            directory = HERE / f"eval{update}_v1" / name
            report = json.loads(bind(directory / "report.json").read_text())
            assert report["identity"]["completed_training_updates"] == update
            reports.append(report)
            traces.append(arrays(directory / "trace.npz", report["trace_sha256"]))
        assert reports[0]["timeline"] == reports[1]["timeline"]
        timeline = reports[0]["timeline"]
        motion = arrays(timeline["timeline_path"], timeline["timeline_sha256"])
        phase = next(r for r in timeline["phases"] if r["name"] == "source_motion")
        for key in ("qpos", "qvel"):
            np.testing.assert_array_equal(traces[0][key][0], traces[1][key][0])
        common = min(len(t["qpos"]) for t in traces) - 1
        row = dict(
            name=name,
            common_controls=common,
            at100=metrics(traces[0], motion, phase, common),
            at500=metrics(traces[1], motion, phase, common),
            entire_source_controls=[r["entire_executed_source"]["source_controls"] for r in reports],
            requested_source_controls=phase["requested_controls"],
            deployment_ready=False,
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = dict(
        kind="existing_pico_declared_milestone_comparison_v1",
        cases=rows,
        inputs=inputs,
        same_initial_physics_and_reference=True,
        same_training_lineage=True,
        same_training_budget=False,
        checkpoint_selection_or_promotion=False,
        deployment_ready=False,
        hardware_authorized=False,
    )
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
