"""Repair small foot-orientation interpolation defects in offline training data.

Only an already accepted control-grid fit with exclusively original-time foot
orientation failures is eligible. Fit local affected-leg knots, then rerun all original
timestamps and the complete serialized control trajectory. No source rescaling,
cropping, limit changes, live control, or dynamic qualification is performed.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation, Slerp

from gear_sonic.utils import g1_23dof_task_space_retarget as ik
from gear_sonic.utils.g1_true23_bones_seed import write_json
from gear_sonic.utils.g1_true23_bones_seed_source_audit import audit_original_timestamps
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_retarget import (
    AdaptationLimits,
    _candidate_assessment,
    _reduce_excursion,
    _refined_result_from_qpos,
    _resample,
    _restore_retained_diagnostic,
    _source_task_poses,
    validate_named_motion,
)
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256

MAX_CORRECTION_RAD = 0.01
FIT_ORIENTATION_MARGIN_RAD = 2e-6
LEG_JOINTS = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")


def repair_coordinates(audit, source_times, control_source_times, joint_names):
    """Select both bracketing knots and one neighboring knot per failed sample."""
    allowed = {f"{foot}_orientation_regression" for foot in ("left_foot", "right_foot")}
    failures = set(audit["failures"])
    if not failures or not failures <= allowed:
        raise ValueError("repair requires only original-time foot-orientation failures")
    coordinates = set()
    for category in failures:
        foot = category.removesuffix("_foot_orientation_regression")
        frames = audit["categories"][category]["frame_indices"]
        if not frames or len(frames) > 32:
            raise ValueError("local repair requires 1..32 failed original frames per foot")
        for frame in frames:
            upper = int(np.searchsorted(control_source_times, source_times[frame], side="right"))
            for knot in range(max(0, upper - 2), min(len(control_source_times), upper + 2)):
                for joint in LEG_JOINTS:
                    coordinates.add((knot, joint_names.index(f"{foot}_{joint}_joint")))
    if not coordinates or len(coordinates) > 128:
        raise ValueError("local repair would require too many control variables")
    return np.asarray(sorted(coordinates), dtype=int)


def _load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def verified_repair_acceptance(original_pose, repaired_pose, coordinates, control_failures, original_audit):
    """Accept independently verified feasibility, never claim solver optimality.

    SLSQP's iteration budget is not a physical limit. A nonconverged iterate may
    be usable only if the complete independent FK/trajectory gates pass and its
    actual serialized edits remain inside the original bounded repair scope.
    """
    before, after = np.asarray(original_pose), np.asarray(repaired_pose)
    if before.shape != after.shape or before.ndim != 2 or before.shape[1] != 30:
        raise ValueError("repair verification requires matching native23 qpos trajectories")
    selected = np.zeros(before.shape, dtype=bool)
    selected[coordinates[:, 0], 7 + coordinates[:, 1]] = True
    checks = {
        "serialized_pose_finite": bool(np.isfinite(after).all()),
        "all_unselected_pose_components_exactly_unchanged": bool(
            np.array_equal(before[~selected], after[~selected])
        ),
        "serialized_correction_within_original_0p01_rad_bound": bool(
            np.max(np.abs(after[selected] - before[selected])) <= MAX_CORRECTION_RAD
        ),
        "complete_control_grid_gates_passed": control_failures == [],
        "all_original_timestamp_gates_passed": (
            original_audit is not None and original_audit.get("original_timestamp_fidelity_passed") is True
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "optimizer_convergence_is_feasibility_evidence": False,
        "minimum_norm_optimality_proven": False,
    }


class _SerializedFeasible(Exception):
    """Bounded search found feasibility; this does not prove optimality."""


class LocalOrientationFit:
    """Finite, bounded knot correction with both original/control FK constraints."""

    def __init__(
        self,
        source_model,
        target_model,
        source,
        candidate,
        pose,
        time_map,
        coordinates,
        config,
        *,
        fit_com_margin_m=0.0,
    ):
        if not np.isfinite(fit_com_margin_m) or not 0 <= fit_com_margin_m <= 1e-6:
            raise ValueError("COM fit margin must tighten the unchanged gate by at most 1e-6 m")
        self.fit_com_margin_m = fit_com_margin_m
        self.source_model, self.target_model = source_model, target_model
        self.layout = ik._model_layout(target_model)
        self.source_layout = ik._model_layout(source_model)
        self.data, self.source_data = mujoco.MjData(target_model), mujoco.MjData(source_model)
        self.pose, self.coordinates, self.config = pose.copy(), coordinates.copy(), config
        self.time_map = time_map
        self.source = source
        self.original_root = np.column_stack(
            [np.interp(source["timestamps_s"], time_map, pose[:, column]) for column in range(3)]
        )
        self.original_quat = Slerp(time_map, Rotation.from_quat(pose[:, [4, 5, 6, 3]]))(
            source["timestamps_s"]
        ).as_quat()[:, [3, 0, 1, 2]]
        changed = np.unique(coordinates[:, 0])
        upper = np.clip(np.searchsorted(time_map, source["timestamps_s"], side="right"), 1, len(time_map) - 1)
        self.affected_original = np.flatnonzero(np.isin(upper, changed) | np.isin(upper - 1, changed))
        self.records = []
        safe = ik._safe_target_layout(self.layout, config.safe_limit_guard_rad, config.native_action_clip)
        mapping = [self.source_layout.joint_names.index(name) for name in self.layout.joint_names]
        for grid, motion, frames in (
            ("control", candidate, changed),
            ("original", source, self.affected_original),
        ):
            for frame in frames:
                ik._set_configuration(
                    source_model,
                    self.source_data,
                    self.source_layout,
                    motion["root_pos_w"][frame],
                    motion["root_quat_wxyz"][frame],
                    motion["joint_pos"][frame],
                )
                targets = ik._task_targets(source_model, self.source_data, ik.DEFAULT_TASKS)
                contacts = tuple(bool(x) for x in motion["contact_flags"][frame])
                direct = np.clip(motion["joint_pos"][frame, mapping], safe.lower, safe.upper)
                baseline = self._measure(
                    motion["root_pos_w"][frame], motion["root_quat_wxyz"][frame], direct, targets, contacts
                )
                self.records.append((grid, int(frame), targets, contacts, baseline))

    def _measure(self, root, quat, joints, targets, contacts):
        ik._set_configuration(self.target_model, self.data, self.layout, root, quat, joints)
        jac, residual, position, orientation, _, _ = ik._task_linearization(
            self.target_model,
            self.data,
            self.layout,
            ik.DEFAULT_TASKS,
            targets,
            contacts,
            self.config.contact_weight_multiplier,
            np.arange(23),
        )
        return ik._weighted_task_error(jac, residual), position, orientation

    def joints(self, delta):
        result = self.pose[:, 7:].copy()
        result[self.coordinates[:, 0], self.coordinates[:, 1]] += delta
        return result

    def constraints(self, delta):
        joints = self.joints(delta)
        original_joints = np.column_stack(
            [np.interp(self.source["timestamps_s"], self.time_map, joints[:, i]) for i in range(23)]
        )
        indices = {task.name: i for i, task in enumerate(ik.DEFAULT_TASKS)}
        margins = []
        cfg = self.config
        for grid, frame, targets, contacts, (before_cost, before_pos, before_ori) in self.records:
            if grid == "control":
                root, quat, joint = self.pose[frame, :3], self.pose[frame, 3:7], joints[frame]
            else:
                root, quat, joint = self.original_root[frame], self.original_quat[frame], original_joints[frame]
            cost, position, orientation = self._measure(root, quat, joint, targets, contacts)
            margins.append(before_cost + cfg.priority_relative_tolerance * max(1, before_cost) - cost)
            com = indices["whole_robot_com"]
            margins.append(
                before_pos[com] + cfg.valid_max_com_regression_m - position[com] - self.fit_com_margin_m
            )
            for foot in ("left_foot", "right_foot"):
                index = indices[foot]
                margins.extend(
                    (
                        cfg.valid_max_foot_position_error_m - position[index],
                        before_ori[index]
                        + cfg.valid_max_foot_orientation_regression_rad
                        - orientation[index]
                        - FIT_ORIENTATION_MARGIN_RAD,
                    )
                )
        velocity, acceleration = ik.trajectory_derivatives(joints, 0.02)
        margins.extend((5.0 - np.abs(velocity)).reshape(-1))
        margins.extend((80.0 - np.abs(acceleration)).reshape(-1))
        return np.asarray(margins)

    def solve(self, *, stop_when_serialized_feasible=False):
        low, high = ik.safe_target_joint_bounds(
            self.target_model, native_action_clip=9.5, safe_limit_guard_rad=0.05
        )
        base = self.pose[self.coordinates[:, 0], 7 + self.coordinates[:, 1]]
        bounds = list(
            zip(
                np.maximum(-MAX_CORRECTION_RAD, low[self.coordinates[:, 1]] - base),
                np.minimum(MAX_CORRECTION_RAD, high[self.coordinates[:, 1]] - base),
                strict=True,
            )
        )
        feasible = None
        callback_count = 0

        def check_serialized(delta):
            nonlocal feasible, callback_count
            callback_count += 1
            serialized = (base + delta).astype(np.float32).astype(float) - base
            if (
                np.isfinite(serialized).all()
                and np.max(np.abs(serialized)) <= MAX_CORRECTION_RAD
                and self.constraints(serialized).min() >= 0
            ):
                feasible = serialized
                raise _SerializedFeasible

        try:
            result = minimize(
                lambda x: float(x @ x / 2),
                np.zeros(len(self.coordinates)),
                jac=lambda x: x,
                method="SLSQP",
                bounds=bounds,
                constraints=[{"type": "ineq", "fun": self.constraints}],
                callback=check_serialized if stop_when_serialized_feasible else None,
                options={"maxiter": 80, "ftol": 1e-13},
            )
            delta = result.x
            success, message, iterations = bool(result.success), str(result.message), int(result.nit)
        except _SerializedFeasible:
            delta = feasible
            success, message, iterations = (
                False,
                "serialized feasible iterate; optimality not sought",
                callback_count,
            )
        pose = self.pose.copy()
        # All downstream evidence is computed from the actual float32 knots.
        pose[:, 7:] = self.joints(delta).astype(np.float32).astype(float)
        actual_delta = pose[self.coordinates[:, 0], 7 + self.coordinates[:, 1]] - base
        return pose, {
            "method": "bounded_slsqp_local_affected_leg_knots_with_both_source_and_control_FK",
            "optimizer_success": success,
            "optimizer_message": message,
            "optimizer_iterations": iterations,
            "max_correction_bound_rad": MAX_CORRECTION_RAD,
            "internal_com_fit_margin_m": self.fit_com_margin_m,
            "independent_acceptance_thresholds_changed": False,
            "serialized_max_correction_rad": float(np.abs(actual_delta).max()),
            "coordinates_frame_joint": self.coordinates.tolist(),
            "affected_original_frames": self.affected_original.tolist(),
            "serialized_selected_constraint_min": float(self.constraints(actual_delta).min()),
            "source_samples_or_root_changed": False,
            "thresholds_changed": False,
            "iterations": [{"accepted": success}],
            "seed_projection": {"iterations": 0},
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--retained-directory", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--grouped-protected-repair",
        action="store_true",
        help="Explicitly allow separated bounded COM/foot-orientation repairs; all original gates still apply",
    )
    args = parser.parse_args(argv)
    output = args.output_directory.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")
    source_dir, retained = args.source_directory.resolve(strict=True), args.retained_directory.resolve(strict=True)
    receipt_path, source_path = source_dir / "report.json", source_dir / "source.named29.npz"
    report_path, adapted_path = retained / "report.json", retained / "adapted.true23.npz"
    diagnostic_path = retained / "fit.diagnostic.npz"
    receipt, parent = json.loads(receipt_path.read_text()), json.loads(report_path.read_text())
    if (
        receipt.get("kind") != "g1_true23_bones_seed_named29_source_v1"
        or receipt["indexed_source"]["split"] != "train"
        or receipt["output"]["sha256"] != sha256_file(source_path)
        or parent.get("bones_seed_source_receipt_sha256") != sha256_file(receipt_path)
        or parent["output"]["sha256"] != sha256_file(adapted_path)
        or parent["diagnostic_output"]["sha256"] != sha256_file(diagnostic_path)
    ):
        raise ValueError("repair requires the intact train-split source and accepted retained fit")
    for path in (source_path, args.source_model, args.target_model):
        if parent["input_bindings"].get(str(path.resolve())) != sha256_file(path):
            raise ValueError("repair source motion or model differs from parent")
    from gear_sonic.utils.g1_true23_generalist_source_closure import collect_local_source_closure

    root = Path(__file__).resolve().parents[2]
    paths = [
        receipt_path,
        source_path,
        report_path,
        adapted_path,
        diagnostic_path,
        args.source_model,
        args.target_model,
    ]
    paths.extend(collect_local_source_closure(root, [Path(__file__)]).files)
    bindings = {str(path.resolve()): sha256_file(path) for path in paths}
    source_model, target_model = ik.load_models(args.source_model, args.target_model)
    arrays, adapted, stored = _load_npz(source_path), _load_npz(adapted_path), _load_npz(diagnostic_path)
    before = audit_original_timestamps(source_model, target_model, arrays, adapted, parent)
    source = validate_named_motion(arrays, source_model, allow_source_limit_excess=True)
    config = ik.RetargetConfig(**parent["ik_config"])
    if "contact_flags" not in source:
        feet = ik._source_foot_positions(
            source_model,
            ik._model_layout(source_model),
            source["root_pos_w"],
            source["root_quat_wxyz"],
            source["joint_pos"],
        )
        source["contact_flags"] = ik.infer_foot_contacts(
            feet,
            fps=float(source["fps"][0]),
            height_tolerance_m=config.contact_height_tolerance_m,
            speed_tolerance_m_s=config.contact_speed_tolerance_m_s,
        )
    declared = deepcopy(parent["limits"])
    for key in ("duration_scales", "excursion_scales"):
        declared[key] = tuple(declared[key])
    limits = AdaptationLimits(**declared)
    attempt = parent["attempts"][parent["selected_attempt"]]
    if attempt["requested_excursion_scale"] != 1.0:
        raise ValueError("local interpolation repair currently requires an unreduced source excursion")
    original, time_map, _ = _resample(source, attempt["requested_duration_scale"], limits)
    candidate = _reduce_excursion(original, 1.0)
    baseline = _restore_retained_diagnostic(source_model, target_model, candidate, stored, config)
    parent_pose = np.column_stack(
        (adapted["body_pos_w"][:, 0], adapted["body_quat_w"][:, 0], adapted["joint_pos"])
    ).astype(float)
    if not np.allclose(parent_pose, stored["diagnostic_qpos_native23"], rtol=0, atol=1e-7):
        raise ValueError("accepted motion differs from retained diagnostic pose")
    coordinate_selector, fit_class = repair_coordinates, LocalOrientationFit
    strategy = "local_original_time_orientation_repair_v1"
    audit_kind = "g1_true23_bones_seed_local_original_time_orientation_repair_v1"
    if args.grouped_protected_repair:
        from gear_sonic.scripts.refine_g1_true23_original_time_protected_groups import (
            GroupedProtectedFit,
            grouped_repair_coordinates,
        )

        coordinate_selector, fit_class = grouped_repair_coordinates, GroupedProtectedFit
        strategy = "grouped_original_time_protected_repair_v1"
        audit_kind = "g1_true23_bones_seed_grouped_original_time_protected_repair_v1"
    coordinates = coordinate_selector(
        before, source["timestamps_s"], time_map, ik._model_layout(target_model).joint_names
    )
    problem = fit_class(source_model, target_model, source, candidate, parent_pose, time_map, coordinates, config)
    pose, solver = problem.solve()
    result = _refined_result_from_qpos(source_model, target_model, candidate, baseline, pose, solver)
    original_positions, _ = _source_task_poses(source_model, original)
    summary, fidelity, failures = _candidate_assessment(result, original_positions, limits)
    rebuilt = {
        **adapted,
        **ik.build_mjlab_motion_arrays(target_model, result),
        "achieved_task_pos_w": result.achieved_task_pos_w,
    }
    report = deepcopy(parent)
    selected = report["attempts"][report["selected_attempt"]]
    selected.update(accepted=not failures, failures=failures, ik_summary=summary, fidelity=fidelity)
    selected.setdefault("solver_attempts", []).append({"strategy": strategy, "fit_report": solver})
    report.update(
        accepted=not failures, input_bindings=bindings, hardware_authorized=False, deployment_ready=False
    )
    report.pop("output", None)
    report.pop("diagnostic_output", None)
    after = (
        audit_original_timestamps(source_model, target_model, arrays, rebuilt, report) if not failures else None
    )
    acceptance = verified_repair_acceptance(parent_pose, pose, coordinates, failures, after)
    passed = acceptance["passed"]
    audit = {
        "kind": audit_kind,
        "kinematic_repair_passed": bool(passed),
        "before": before,
        "after": after,
        "solver": solver,
        "independent_serialized_acceptance": acceptance,
        "control_grid_failures": failures,
        "source_frame_count_unchanged": len(source["joint_pos"]),
        "control_frame_count_unchanged": len(pose),
        "input_bindings": bindings,
        "compiled_models": {
            "source": compiled_model_sha256(source_model),
            "target": compiled_model_sha256(target_model),
        },
        "dynamic_feasibility_verified": False,
        "training_corpus_ready": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
    for path, expected in bindings.items():
        if sha256_file(Path(path)) != expected:
            raise ValueError(f"repair input changed during fit: {path}")
    output.mkdir(parents=True, exist_ok=False)
    if passed:
        destination = output / "adapted.true23.npz"
        with destination.open("xb") as stream:
            np.savez_compressed(stream, **rebuilt)
        report["output"] = {"path": destination.name, "sha256": sha256_file(destination)}
    else:
        report["accepted"] = False
        report["selected_attempt"] = None
        destination = output / "repair.rejected.npz"
        with destination.open("xb") as stream:
            np.savez_compressed(stream, diagnostic_only_not_accepted_motion=np.array([True]), qpos_native23=pose)
        audit["rejected_candidate"] = {"path": destination.name, "sha256": sha256_file(destination)}
    write_json(output / "report.json", report)
    write_json(output / "original_time_repair.json", audit)
    print(
        json.dumps(
            {
                "path": str(output),
                "kinematic_repair_passed": bool(passed),
                "control_grid_failures": failures,
                "original_time_failures": after["failures"] if after else None,
                "max_joint_correction_rad": solver["serialized_max_correction_rad"],
            }
        ),
        flush=True,
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
