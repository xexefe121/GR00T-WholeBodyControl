"""Read-only phase diagnosis of completed CPU traces; never shifts scored data."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def measure_phase(trace, reference_joints, start, stop, *, max_lag=20):
    measured = np.asarray(trace["qpos"])[1:, 7:]
    reference = np.asarray(reference_joints)[11 : 11 + len(measured)]
    if (
        measured.ndim != 2
        or measured.shape != reference.shape
        or measured.shape[1] != 23
        or not 0 <= start < stop <= len(measured)
        or stop - start <= 2 * max_lag
        or not np.isfinite(measured).all()
        or not np.isfinite(reference).all()
    ):
        raise ValueError("requires complete finite native23 source phase, not a prefix")
    error = measured[start:stop] - reference[start:stop]
    interior = slice(start + max_lag, stop - max_lag)
    # Diagnostic correlation uses an interior window. It does not overwrite the
    # full unshifted error above, referee thresholds, or original case outcome.
    lag_scores = [
        float(
            np.sqrt(
                np.mean(
                    (measured[interior, :12] - reference[start + max_lag + lag : stop - max_lag + lag, :12]) ** 2
                )
            )
        )
        for lag in range(-max_lag, max_lag + 1)
    ]
    best = int(np.argmin(lag_scores)) - max_lag
    result = dict(
        source_controls=[start, stop],
        full_source_control_count=stop - start,
        all_joint_rmse_rad=float(np.sqrt(np.mean(error**2))),
        leg_rmse_rad=float(np.sqrt(np.mean(error[:, :12] ** 2))),
        per_hardware_joint_rmse_rad=dict(
            zip(HARDWARE_23_JOINT_NAMES, np.sqrt(np.mean(error**2, axis=0)).tolist())
        ),
        lag_diagnostic=dict(
            interior_controls=[interior.start, interior.stop],
            max_search_lag_frames=max_lag,
            unshifted_interior_leg_rmse_rad=lag_scores[max_lag],
            best_reference_offset_frames=best,
            estimated_measured_leg_lag_s=-best * 0.02,
            shifted_interior_leg_rmse_rad=min(lag_scores),
            reference_or_accepted_metrics_time_shifted=False,
            qualification_evidence=False,
        ),
        relative_landmark_p95_m=np.percentile(trace["relative_landmark_error_m"][start:stop], 95, axis=0).tolist(),
        relative_landmark_order=["left_ankle", "right_ankle", "left_hand", "right_hand", "head"],
        deployment_ready=False,
        hardware_authorized=False,
    )
    projection = trace.get("bounded_linear_projection_delta_rad")
    if projection is not None:
        result["source_controls_with_any_projected_action_fraction"] = float(
            np.mean(np.any(np.abs(projection[start:stop]) > 1e-7, axis=1))
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report_path = args.evaluation_directory.resolve(strict=True) / "report.json"
    bindings = {str(report_path): digest(report_path), str(Path(__file__).resolve()): digest(__file__)}
    report = json.loads(report_path.read_text())
    reference_path = Path(report["timeline"]["timeline_path"])
    bindings[str(reference_path)] = report["timeline"]["timeline_sha256"]
    if digest(reference_path) != bindings[str(reference_path)]:
        raise ValueError("reference hash mismatch")
    with np.load(reference_path, allow_pickle=False) as data:
        reference = data["joint_pos"].copy()
    source = next(phase for phase in report["timeline"]["phases"] if phase["name"] == "source_motion")
    rows = []
    for case in report["records"]:
        trace_path = Path(case["trace_path"])
        bindings[str(trace_path)] = case["trace_sha256"]
        if digest(trace_path) != case["trace_sha256"]:
            raise ValueError("trace hash mismatch")
        with np.load(trace_path, allow_pickle=False) as data:
            trace = {
                name: data[name].copy()
                for name in ("qpos", "relative_landmark_error_m", "bounded_linear_projection_delta_rad")
                if name in data
            }
        rows.append(
            dict(
                case=case["case"],
                policy_source=case["policy_identity"]["source"],
                original_tracking_result=case["lifecycle"]["source_motion_tracking"],
                metrics=measure_phase(trace, reference, source["control_start"], source["control_stop"]),
            )
        )
    for path, expected in bindings.items():
        if digest(path) != expected:
            raise ValueError("phase measurement input changed")
    result = dict(
        kind="g1_true23_saved_trace_unshifted_tracking_and_lag_diagnostic_v1",
        inputs=bindings,
        cases=rows,
        reference_acceptance_rules_modified=False,
        simulation_rerun=False,
        hardware_authorized=False,
        deployment_ready=False,
    )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cases": [
                    dict(
                        case=row["case"],
                        leg_rmse=row["metrics"]["leg_rmse_rad"],
                        lag_s=row["metrics"]["lag_diagnostic"]["estimated_measured_leg_lag_s"],
                    )
                    for row in rows
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
