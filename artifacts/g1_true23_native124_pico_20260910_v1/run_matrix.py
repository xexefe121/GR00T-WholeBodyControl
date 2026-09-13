"""Fixed full-lifecycle matrix of selected native124; never a SONIC promotion."""

import json
from pathlib import Path
import sys

import numpy as np

from gear_sonic.utils.g1_23dof_native124_21204_adapter import (
    Native124Checkpoint21204Policy,
    load_checkpoint21204_binding,
)
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS, run_reference_diagnostic
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_lifecycle import assess_lifecycle_diagnostic
from gear_sonic.utils.g1_true23_native124_pico_comparison import Native124PicoComparator, PHASES

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT.parent / "GR00T-WholeBodyControl"
BASE = Path("/mnt/c/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1")
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "actual_v1"
FLAGS = dict(
    hardware_authorized=False, deployment_ready=False, teacher_label_admitted=False, policy_is_sonic=False
)


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def main():
    if OUTPUT.exists():
        raise FileExistsError("fixed matrix refuses overwrite or implicit retry")
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = sha256_file(path)
        if expected is not None and digest != expected:
            raise ValueError("input hash mismatch: " + str(path))
        inputs[str(path)] = digest
        return path

    binding = load_checkpoint21204_binding(ASSETS)
    for name in (
        "manifest_path",
        "selection_path",
        "export_report_path",
        "checkpoint_path",
        "onnx_path",
        "resolved_env_evidence_path",
    ):
        bind(getattr(binding, name))
    bind(ASSETS / MODEL, "16e304c970bfc68783ea69d01d05192e6bd9d83d62f6ee4aac0ac72ff18db612")
    bind(ROOT / PHYSICS, "bc4dab246e709604f435dea7c584adc01f1c38a6abb6ac1bc22fc699c42272c3")
    bind(__file__)
    bind(HERE / "EXPERIMENT.md")
    cases = {}
    for name in ("walk002", "walk003", "walk008", "pico"):
        previous = BASE / "normal_core_adaptation_v1/cpu100_v2" / name / "report.json"
        if name == "pico":
            previous = BASE / "normal_core_pico_v1/report.json"
        old = json.loads(bind(previous).read_text())
        timeline = old["timeline"]
        path = bind(timeline["timeline_path"], timeline["timeline_sha256"])
        with np.load(path, allow_pickle=False) as z:
            motion = {k: z[k].copy() for k in z.files}
        assert len(motion["joint_pos"]) - 11 == timeline["total_requested_controls"]
        cases[name] = dict(path=path, motion=motion, timeline=timeline)
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("gear_sonic.") and path and str(path).endswith(".py"):
            bind(path)
    OUTPUT.mkdir()
    write(OUTPUT / "started.json", dict(inputs=inputs, cases=list(cases), phases=PHASES, **FLAGS))
    summaries = []
    for phase in PHASES:
        for name, case in cases.items():
            directory = OUTPUT / (phase + "_" + name)
            directory.mkdir()
            write(
                directory / "request.json",
                dict(
                    name=name,
                    phase=phase,
                    requested_controls=case["timeline"]["total_requested_controls"],
                    inputs=inputs,
                    **FLAGS,
                ),
            )
            print(json.dumps(dict(starting=name, phase=phase, full_lifecycle_requested=True)), flush=True)
            policy = Native124Checkpoint21204Policy(binding)
            adapter = Native124PicoComparator(case["motion"], phase=phase, root=ROOT, assets=ASSETS)
            result, trace = run_reference_diagnostic(
                root=ROOT, asset_root=ASSETS, motion_path=case["path"], policy=policy, runtime_adapter=adapter
            )
            attempts = adapter.arrays()
            for filename, values in (("trace.npz", trace), ("attempts.npz", attempts)):
                with (directory / filename).open("xb") as stream:
                    np.savez_compressed(stream, **values)
            lifecycle = assess_lifecycle_diagnostic(case["timeline"], result, trace)
            phase_row = next(row for row in lifecycle["phases"] if row["name"] == "source_motion")
            start, end = phase_row["control_start"], min(result["completed_controls"], phase_row["control_stop"])
            metrics = {}
            if end > start:
                delta = trace["qpos"][start + 1 : end + 1, 7:] - case["motion"]["joint_pos"][start + 11 : end + 11]
                metrics = dict(
                    source_controls=end - start,
                    leg_rmse_rad=float(np.sqrt(np.mean(delta[:, :12] ** 2))),
                    arm_rmse_rad=float(np.sqrt(np.mean(delta[:, 13:] ** 2))),
                    root_p95_m=float(np.percentile(trace["pelvis_error_m"][start:end], 95)),
                    relative_foot_p95_m=np.percentile(
                        trace["relative_landmark_error_m"][start:end, :2], 95, axis=0
                    ).tolist(),
                )
            report = dict(
                name=name,
                phase=phase,
                result=result,
                lifecycle=lifecycle,
                timeline=case["timeline"],
                source_metrics=metrics,
                trace_sha256=sha256_file(directory / "trace.npz"),
                attempts_sha256=sha256_file(directory / "attempts.npz"),
                inputs=inputs,
                **FLAGS,
            )
            for path, expected in inputs.items():
                assert sha256_file(path) == expected, path
            write(directory / "report.json", report)
            summary = dict(
                case=directory.name,
                completed_controls=result["completed_controls"],
                requested_controls=result["requested_controls"],
                failure=result["failure"],
                metrics=metrics,
                source_tracking_passed=lifecycle["source_motion_tracking"][
                    "provisional_reference_landmark_screen_passed"
                ],
                hard_range_excess_rad=result["hard_joint_limit_excess_max_rad"],
                velocity_ratio=result["joint_velocity_ratio_max"],
                report_sha256=sha256_file(directory / "report.json"),
                **FLAGS,
            )
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
    write(OUTPUT / "summary.json", dict(cases=summaries, all_eight_requested=True, **FLAGS))


if __name__ == "__main__":
    main()
