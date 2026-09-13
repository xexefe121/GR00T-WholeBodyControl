"""Offline reference torque-authority audit, NEVER a controller or SIM pass.

Compare identical candidate contact cones and optimistic bounded friction under
motor-only limits versus the actual reachable bounded-position PD torque range.
Every commanded reference uses the same received three-pose derivative window
as the diagnostic controller. No simulator stepping or hardware transport.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import scipy
from scipy.optimize import linprog

from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_source_closure import collect_local_source_closure
from gear_sonic.utils.g1_true23_inverse_dynamics_tracker import InverseDynamicsTracker
from gear_sonic.utils.g1_true23_native_model_actuation import NativeModelActuationProfile
from gear_sonic.utils.g1_true23_reference_floor import motion_qpos
from gear_sonic.utils.g1_true23_reference_support import floor_contact_map, pose_path_derivatives
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model


def bounded_force_support(required, contact_map, lower, upper, friction):
    """Conditional LP feasibility; unknown is NEVER counted as infeasible."""
    required, contact_map, lower, upper, friction = (
        np.asarray(value, dtype=np.float64) for value in (required, contact_map, lower, upper, friction)
    )
    if (
        required.shape != (29,)
        or contact_map.ndim != 2
        or contact_map.shape[0] != 29
        or any(value.shape != (23,) for value in (lower, upper, friction))
        or any(not np.isfinite(value).all() for value in (required, contact_map, lower, upper, friction))
        or np.any(friction < 0)
    ):
        raise ValueError("authority LP requires finite native23 inputs and nonnegative friction bounds")
    if np.any(lower > upper):
        return dict(status="infeasible", reason="empty_reachable_motor_torque_interval", solver_status=None)
    rays = contact_map.shape[1]
    joint_map = np.vstack((np.zeros((6, 23)), np.eye(23)))
    equality = np.column_stack((contact_map, joint_map, joint_map))
    bounds = (
        [(0, None)] * rays + list(zip(lower, upper, strict=True)) + list(zip(-friction, friction, strict=True))
    )
    options = dict(primal_feasibility_tolerance=1e-8, dual_feasibility_tolerance=1e-8)
    result = linprog(
        np.zeros(rays + 46), A_eq=equality, b_eq=required, bounds=bounds, method="highs", options=options
    )
    retried = result.status not in (0, 2)
    if retried:
        result = linprog(
            np.zeros(rays + 46),
            A_eq=equality,
            b_eq=required,
            bounds=bounds,
            method="highs-ds",
            options={**options, "presolve": False},
        )
    record = dict(solver_status=int(result.status), retried_identical_lp_without_presolve=retried)
    if result.status != 0:
        return {
            **record,
            "status": "infeasible" if result.status == 2 else "unknown",
            "reason": str(result.message),
        }
    solution = np.asarray(result.x)
    if solution.shape != (rays + 46,) or not np.isfinite(solution).all():
        raise RuntimeError("authority LP returned invalid supposedly successful solution")
    torque, assistance = solution[rays : rays + 23], solution[rays + 23 :]
    residual = float(np.max(np.abs(equality @ solution - required)))
    bound_error = max(
        np.maximum(-solution[:rays], 0).max(initial=0),
        np.maximum(lower - torque, 0).max(initial=0),
        np.maximum(torque - upper, 0).max(initial=0),
        np.maximum(np.abs(assistance) - friction, 0).max(initial=0),
    )
    if residual > 1e-5 or bound_error > 1e-7:
        raise RuntimeError("authority LP failed independent original force or bound audit")
    return {
        **record,
        "status": "conditional_feasible",
        "maximum_force_residual": residual,
        "maximum_bound_violation": float(bound_error),
        "joint_torque23_nm": torque.tolist(),
        "optimistic_friction23_nm": assistance.tolist(),
        "candidate_ray_weights": solution[:rays].tolist(),
    }


def reconcile_verified_witnesses(motor, pd, lower, upper, effort):
    """A verified feasible point also certifies any containing torque envelope.

    LP status tolerances and the independently audited force tolerance differ.
    Preserve raw solver outcomes, but do not report a negative status when the
    paired solve supplies a point satisfying the SAME acceptance checks.
    This does not relax either force or torque-bound acceptance tolerance.
    """
    pairs = [dict(motor), dict(pd)]
    envelopes = [(-np.asarray(effort), np.asarray(effort)), (np.asarray(lower), np.asarray(upper))]
    for source, destination in ((0, 1), (1, 0)):
        witness, current = pairs[source], pairs[destination]
        if witness["status"] != "conditional_feasible" or current["status"] == "conditional_feasible":
            continue
        torque = np.asarray(witness["joint_torque23_nm"])
        lo, hi = envelopes[destination]
        if (
            torque.shape != (23,)
            or not np.isfinite(torque).all()
            or not np.isfinite(witness["maximum_force_residual"])
            or not np.isfinite(witness["maximum_bound_violation"])
            or witness["maximum_force_residual"] > 1e-5
            or witness["maximum_bound_violation"] > 1e-7
        ):
            raise ValueError("authority reconciliation requires an independently verified witness")
        if np.any(lo > hi) or np.any(torque < lo - 1e-7) or np.any(torque > hi + 1e-7):
            continue
        pairs[destination] = {
            **witness,
            "original_destination_solver_result": current,
            "feasibility_witness_from_paired_envelope": "motor" if source == 0 else "pd",
        }
    return tuple(pairs)


def summarize_rows(rows):
    return dict(
        reference_controls=len(rows),
        motor_conditionally_feasible=sum(row["motor"]["status"] == "conditional_feasible" for row in rows),
        pd_conditionally_feasible=sum(row["pd"]["status"] == "conditional_feasible" for row in rows),
        motor_infeasible=sum(row["motor"]["status"] == "infeasible" for row in rows),
        pd_infeasible=sum(row["pd"]["status"] == "infeasible" for row in rows),
        motor_unknown=sum(row["motor"]["status"] == "unknown" for row in rows),
        pd_unknown=sum(row["pd"]["status"] == "unknown" for row in rows),
        motor_feasible_but_pd_infeasible=sum(
            row["motor"]["status"] == "conditional_feasible" and row["pd"]["status"] == "infeasible"
            for row in rows
        ),
        paired_feasibility_witnesses_used=sum(
            "feasibility_witness_from_paired_envelope" in row[kind] for row in rows for kind in ("motor", "pd")
        ),
        dynamic_feasibility_proven=False,
        simulator_qualified=False,
        hardware_authorized=False,
        deployment_ready=False,
    )


def configure_candidate_gap(tracker, gap_m):
    """Change an OFFLINE contact hypothesis only, never the physical model."""
    if gap_m not in (0.004, 0.005):
        raise ValueError("authority sensitivity permits only declared 4 or 5 mm candidates")
    tracker.model.geom_margin[:] = np.maximum(tracker.model.geom_margin, gap_m)
    tracker.model.pair_margin[:] = np.maximum(tracker.model.pair_margin, gap_m)
    tracker.assert_original_unchanged()


def pose_time_derivatives(model, window, duration_scale):
    """Offline uniform-time hypothesis at original pose samples, not a replay."""
    if duration_scale not in (1.0, 2.0, 4.0):
        raise ValueError("authority time sensitivity permits only declared 1x, 2x or 4x duration")
    return pose_path_derivatives(model, window, 0.02 * duration_scale)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-directory", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--candidate-floor-gap-m", type=float, choices=(0.004, 0.005), default=0.004)
    parser.add_argument("--pose-duration-scale", type=float, choices=(1.0, 2.0, 4.0), default=1.0)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    report_path = (args.evaluation_directory / "report.json").resolve(strict=True)
    evaluation = json.loads(report_path.read_text())
    row = next(record for record in evaluation["records"] if record["case"] == "nominal")
    source = Path(row["result"]["motion_path"]).resolve(strict=True)
    if sha256_file(source) != row["result"]["motion_sha256"]:
        raise ValueError("bound full lifecycle reference changed")
    count = int(row["result"]["available_controls"])
    if count != evaluation["timeline"]["total_requested_controls"]:
        raise ValueError("authority audit requires the complete unchanged reference timeline")
    _, model, _ = prepare_true23_model(args.asset_root / MODEL, root / PHYSICS)
    profile = NativeModelActuationProfile.from_sim_config(root / PHYSICS)
    tracker = InverseDynamicsTracker(model, profile)  # Private geometry and unchanged target bounds ONLY.
    configure_candidate_gap(tracker, args.candidate_floor_gap_m)
    if tracker.original_sha != row["result"]["compiled_model_sha256"]:
        raise ValueError("authority audit must use identical native23 physics")
    with np.load(source, allow_pickle=False) as archive:
        poses = motion_qpos(model, {key: archive[key].copy() for key in archive.files})
    if poses.shape != (count + 11, 30) or not np.isfinite(poses).all():
        raise ValueError("authority audit requires every native23 reference frame")
    paths = [
        report_path,
        source,
        args.asset_root / MODEL,
        root / PHYSICS,
        *collect_local_source_closure(root, [Path(__file__)]).files,
    ]
    bindings = {str(path.resolve()): sha256_file(path) for path in paths}
    args.output_directory.mkdir(parents=True, exist_ok=False)
    private, data = tracker.model, tracker.data
    records = []
    for i in range(count):
        frame = i + 10
        velocity, acceleration = pose_time_derivatives(
            private, poses[frame - 1 : frame + 2], args.pose_duration_scale
        )
        data.qpos[:], data.qvel[:] = poses[frame], velocity[1]
        mujoco.mj_fwdPosition(private, data)
        mujoco.mj_fwdVelocity(private, data)
        contact_map, contacts = floor_contact_map(private, data, tracker.plane, args.candidate_floor_gap_m)
        inertial = np.zeros(29)
        mujoco.mj_mulM(private, data, inertial, acceleration[1])
        required = inertial + data.qfrc_bias - data.qfrc_passive
        lower = np.maximum(
            -tracker.effort, tracker.kp * (tracker.target_lower - data.qpos[7:]) - tracker.kd * data.qvel[6:]
        )
        upper = np.minimum(
            tracker.effort, tracker.kp * (tracker.target_upper - data.qpos[7:]) - tracker.kd * data.qvel[6:]
        )
        phase = next(
            p["name"] for p in evaluation["timeline"]["phases"] if p["control_start"] <= i < p["control_stop"]
        )
        records.append(
            dict(
                control_index=i,
                reference_frame=frame,
                control_phase=phase,
                candidate_contacts=len(contacts),
                pd_torque_lower23_nm=lower.tolist(),
                pd_torque_upper23_nm=upper.tolist(),
                motor=bounded_force_support(
                    required, contact_map, -tracker.effort, tracker.effort, private.dof_frictionloss[6:]
                ),
                pd=bounded_force_support(required, contact_map, lower, upper, private.dof_frictionloss[6:]),
            )
        )
        records[-1]["motor"], records[-1]["pd"] = reconcile_verified_witnesses(
            records[-1]["motor"], records[-1]["pd"], lower, upper, tracker.effort
        )
        if (i + 1) % 500 == 0 or i + 1 == count:
            print(
                json.dumps(dict(completed_reference_controls=i + 1, requested_reference_controls=count)),
                flush=True,
            )
    tracker.assert_original_unchanged()
    if any(sha256_file(Path(path)) != digest for path, digest in bindings.items()):
        raise ValueError("authority audit inputs or implementation changed during audit")
    summary = summarize_rows(records)
    report = dict(
        kind="g1_true23_received_reference_pd_torque_authority_audit_v4",
        summary=summary,
        phases={
            p["name"]: summarize_rows([r for r in records if r["control_phase"] == p["name"]])
            for p in evaluation["timeline"]["phases"]
        },
        records=records,
        inputs=bindings,
        physical_compiled_sha256=tracker.original_sha,
        actuation_profile=profile.contract(),
        command_law_audited="original_bounded_target_PD_without_feedforward",
        reference_derivatives="each_original_three_pose_window_at_20ms_times_declared_duration_scale",
        reference_ages_ms=[200, 180, 160] if args.pose_duration_scale == 1 else None,
        pose_duration_scale=args.pose_duration_scale,
        duration_scaling_is_only_offline_hypothesis_not_a_resampled_50hz_rollout=True,
        retimed_between_sample_geometry_or_dynamics_tested=False,
        candidate_floor_gap_m=args.candidate_floor_gap_m,
        contact_hypothesis_sensitivity_is_not_a_change_to_rollout_physics=True,
        may_be_used_as_automatic_dataset_rejection=False,
        same_candidate_cones_and_optimistic_bounded_friction_for_both_LPs=True,
        actual_contacts_or_force_complementarity_proven=False,
        model_was_stepped=False,
        note=(
            "Conditional exact-reference force screen, not a rollout pass "
            "or proof that approximate motion is impossible."
        ),
        versions=dict(mujoco=mujoco.__version__, numpy=np.__version__, scipy=scipy.__version__),
        simulator_qualified=False,
        hardware_authorized=False,
        deployment_ready=False,
    )
    with (args.output_directory / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
