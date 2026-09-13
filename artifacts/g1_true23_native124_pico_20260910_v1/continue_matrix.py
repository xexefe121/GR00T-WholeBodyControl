"""Preserve first rollout and finish only the seven never-run matrix cases.

The original driver saved first-case arrays, then its hash recheck passed a
string instead of Path. No controller rerun is needed or allowed to repair
that report-writer error. Recover the terminal from the saved physical trace.
"""

import json
from pathlib import Path

import numpy as np

import run_matrix as original
from gear_sonic.utils.g1_true23_generalist_benchmark import summarize_tracking
from gear_sonic.utils.g1_true23_step1b_mujoco import _projected_gravity


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def main():
    root, output, base = original.ROOT, original.OUTPUT, original.BASE
    inputs = json.loads((output / "started.json").read_text())["inputs"]
    for path, expected in inputs.items():
        assert original.sha256_file(Path(path)) == expected, path
    inputs[str(Path(__file__).resolve())] = original.sha256_file(Path(__file__))
    original.write(
        output / "continuation.json",
        dict(
            cause="post-rollout report writer passed str instead of Path to sha256_file",
            first_saved_rollout_reused_without_rerun=True,
            remaining_cases=7,
            inputs=inputs,
            **original.FLAGS,
        ),
    )
    binding = original.load_checkpoint21204_binding(original.ASSETS)
    summaries = []
    for phase in original.PHASES:
        for name in ("walk002", "walk003", "walk008", "pico"):
            old_path = base / "normal_core_adaptation_v1/cpu100_v2" / name / "report.json"
            if name == "pico":
                old_path = base / "normal_core_pico_v1/report.json"
            old = json.loads(old_path.read_text())
            timeline = old["timeline"]
            path = Path(timeline["timeline_path"])
            motion = arrays(path)
            directory = output / (phase + "_" + name)
            recovered = phase == "causal_q9" and name == "walk002"
            if recovered:
                assert directory.is_dir() and not (directory / "report.json").exists()
                trace = arrays(directory / "trace.npz")
                attempts = arrays(directory / "attempts.npz")
                n = len(trace["qpos"]) - 1
                assert n == 524 and len(trace["physics_time"]) == n * 10
                assert len(attempts["applied_target23"]) == n
                tilts = np.array(
                    [np.arccos(np.clip(-_projected_gravity(q[3:7])[2], -1, 1)) for q in trace["qpos"][1:]]
                )
                stop = (trace["qpos"][1:, 2] < 0.12) | (tilts > 2.2)
                assert stop[-1] and not stop[:-1].any()
                result = dict(
                    kind="saved_arrays_reconstructed_terminal_v1",
                    completed_controls=n,
                    available_controls=timeline["total_requested_controls"],
                    requested_controls=timeline["total_requested_controls"],
                    failure=dict(
                        type="RuntimeError",
                        message="absolute height/tilt diagnostic stop",
                        completed_controls=n,
                        physics_steps=n * 10,
                    ),
                    compiled_model_sha256="80c82f5374bed69423580b4d2085e2103bfd83b8691dfe1f7699d2434e1561a8",
                    hard_joint_limit_excess_max_rad=float(np.max(trace["physics_hard_limit_excess23"])),
                    joint_velocity_ratio_max=float(np.max(trace["physics_velocity_ratio23"])),
                    original_terminal_reconstructed_not_retrieved=True,
                    saved_terminal_height_m=float(trace["qpos"][-1, 2]),
                    saved_terminal_tilt_rad=float(tilts[-1]),
                )
            else:
                if directory.exists():
                    raise FileExistsError("continuation refuses to repeat any existing case")
                directory.mkdir()
                original.write(
                    directory / "request.json",
                    dict(
                        name=name,
                        phase=phase,
                        requested_controls=timeline["total_requested_controls"],
                        inputs=inputs,
                        **original.FLAGS,
                    ),
                )
                print(json.dumps(dict(starting=name, phase=phase)), flush=True)
                policy = original.Native124Checkpoint21204Policy(binding)
                adapter = original.Native124PicoComparator(motion, phase=phase, root=root, assets=original.ASSETS)
                result, trace = original.run_reference_diagnostic(
                    root=root, asset_root=original.ASSETS, motion_path=path, policy=policy, runtime_adapter=adapter
                )
                # Persist terminal before any report aggregation or hashing.
                original.write(directory / "physics_result.json", result)
                attempts = adapter.arrays()
                for filename, value in (("trace.npz", trace), ("attempts.npz", attempts)):
                    with (directory / filename).open("xb") as stream:
                        np.savez_compressed(stream, **value)
            lifecycle = original.assess_lifecycle_diagnostic(timeline, result, trace)
            source = next(row for row in timeline["phases"] if row["name"] == "source_motion")
            start, end = source["control_start"], min(result["completed_controls"], source["control_stop"])
            metrics = {}
            if end > start:
                delta = trace["qpos"][start + 1 : end + 1, 7:] - motion["joint_pos"][start + 11 : end + 11]
                metrics = dict(
                    source_controls=end - start,
                    leg_rmse_rad=float(np.sqrt(np.mean(delta[:, :12] ** 2))),
                    arm_rmse_rad=float(np.sqrt(np.mean(delta[:, 13:] ** 2))),
                    root_p95_m=float(np.percentile(trace["pelvis_error_m"][start:end], 95)),
                    relative_foot_p95_m=np.percentile(
                        trace["relative_landmark_error_m"][start:end, :2], 95, axis=0
                    ).tolist(),
                )
            full_tracking = summarize_tracking(
                trace["landmark_error_m"],
                completed=result["completed_controls"],
                available=result["available_controls"],
                requested=result["requested_controls"],
                failure=result["failure"],
            )
            report = dict(
                name=name,
                phase=phase,
                result=result,
                lifecycle=lifecycle,
                timeline=timeline,
                source_metrics=metrics,
                full_lifecycle_tracking=full_tracking,
                recovered_saved_arrays_without_rerun=recovered,
                trace_sha256=original.sha256_file(directory / "trace.npz"),
                attempts_sha256=original.sha256_file(directory / "attempts.npz"),
                inputs=inputs,
                **original.FLAGS,
            )
            for pinned, expected in inputs.items():
                assert original.sha256_file(Path(pinned)) == expected, pinned
            original.write(directory / "report.json", report)
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
                report_sha256=original.sha256_file(directory / "report.json"),
                **original.FLAGS,
            )
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
    original.write(
        output / "summary.json",
        dict(
            cases=summaries,
            all_eight_requested=True,
            first_case_report_recovered_without_rerun=True,
            **original.FLAGS,
        ),
    )


if __name__ == "__main__":
    main()
