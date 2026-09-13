"""Bounded multistart pose feasibility diagnostic, never a motion or policy.

Tests whether the worst remaining collision can be cleared with the original
per-frame task/COM/foot and safe joint/root bounds. Temporal constraints are NOT
tested here. Even a successful pose is not a usable trajectory or training clip.
No robot interfaces, dynamics integration, altered collision geometry or gates.
"""

import argparse
from dataclasses import replace
from itertools import product
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import Bounds, minimize

from gear_sonic.scripts.audit_g1_true23_reference_bank_self_contacts import measure_self_contacts
from gear_sonic.scripts.refine_g1_true23_collision_clearance import load_arrays
from gear_sonic.utils import g1_23dof_task_space_retarget as ik
from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_protected_root import audit_norms, residual_groups
from gear_sonic.utils.g1_true23_generalist_retarget import _restore_retained_diagnostic
from gear_sonic.utils.g1_true23_generalist_source_closure import collect_local_source_closure
from gear_sonic.utils.g1_true23_original_task_trajectory import TASK_SCALES, OriginalTaskConfig, OriginalTaskPath
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256
from gear_sonic.utils.g1_true23_self_collision import SelfCollisionLinearizer
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model


class PoseEscape:
    def __init__(
        self, problem, baseline, physical_model, *, upper_position_limit_m=None, interior_solver_margin=False
    ):
        self.problem = problem
        self.numerical_margin_fraction = 0.995 if interior_solver_margin else 1.0
        all_groups, self.width = residual_groups(problem, baseline)
        self.groups = [group for group in all_groups if group["frame"] == 0]
        if upper_position_limit_m is not None:
            if not np.isfinite(upper_position_limit_m) or not 0 < upper_position_limit_m <= 0.1:
                raise ValueError("pose diagnostic upper-position limit must stay within the existing 10-cm gate")
            cursor = 0
            for task in problem.tasks:
                for kind, scale in zip(("position", "orientation"), TASK_SCALES[task.name], strict=True):
                    if scale:
                        if kind == "position" and task.name in {"left_hand", "right_hand", "head_proxy"}:
                            self.groups.append(
                                {
                                    "name": task.name + "_pose_position",
                                    "frame": 0,
                                    "indices": np.arange(cursor, cursor + 3),
                                    "scales": np.full(3, 1 / scale),
                                    "radius": float(upper_position_limit_m),
                                }
                            )
                        cursor += 3
        self.query = SelfCollisionLinearizer(physical_model)
        self.last_x = self.cached = None
        self.last_optimization = None

    def evaluate(self, x, *, optimization=True):
        if self.last_x is not None and optimization == self.last_optimization and np.array_equal(x, self.last_x):
            return self.cached
        margin = self.numerical_margin_fraction if optimization else 1.0
        repeated = np.repeat(np.asarray(x)[None], 3, axis=0)
        residual, jac, _ = self.problem.evaluate(repeated)
        residual, jac = residual[: self.width], jac[: self.width, :29].toarray()
        values, gradients = [], []
        for group in self.groups:
            indices, scales, radius = group["indices"], group["scales"], group["radius"] * margin
            r = residual[indices] * scales
            normalization = max(radius, 1e-12)
            # Squared norm has the identical feasible set without a cusp when
            # an already-fitted foot residual is very close to zero.
            values.append((radius**2 - r @ r) / normalization**2)
            gradients.append(-2 * (r @ (scales[:, None] * jac[indices])) / normalization**2)
        # Eight exact linear facets avoid a nondifferentiable abs() constraint
        # at zero attitude coordinates. They define the identical L1 ball.
        for facet in product((-1.0, 1.0), repeat=3):
            attitude = np.zeros(29)
            attitude[3:6] = -np.asarray(facet) / (self.problem.config.maximum_root_rotation_l1_rad * margin)
            values.append(1 + attitude @ x)
            gradients.append(attitude)
        contacts = self.query.pose_rows(self.problem.qpos(repeated)[0], self.problem.layout.dof_addresses)
        worst = min(contacts, key=lambda row: row["distance_m"], default=None)
        distance = worst["distance_m"] if worst is not None else 0.03
        values.append((distance - 0.001 / margin) / 0.1)
        gradients.append(np.r_[np.zeros(6), worst["joint_jacobian"] / 0.1] if worst is not None else np.zeros(29))
        self.last_x = np.array(x, copy=True)
        self.last_optimization = optimization
        self.cached = np.asarray(values), np.asarray(gradients), residual, worst
        return self.cached


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "candidate-audit",
        "parent-directory",
        "source-model",
        "target-model",
        "asset-root",
        "output-directory",
    ):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--frame", type=int)
    parser.add_argument("--upper-position-limit-m", type=float)
    parser.add_argument("--interior-solver-margin", action="store_true")
    args = parser.parse_args(argv)
    output = args.output_directory.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError("pose diagnostic refuses overwrite")
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = sha256_file(path)
        if expected is not None and digest != expected:
            raise ValueError(f"pose diagnostic binding changed: {path}")
        inputs[str(path)] = digest
        return path

    audit_path = bind(args.candidate_audit)
    audit = json.loads(audit_path.read_text())
    if (
        audit.get("kind") != "g1_true23_full_collision_repair_audit_v1"
        or audit.get("acceptance", {}).get("passed") is not False
    ):
        raise ValueError("pose diagnostic requires retained rejected collision evidence")
    parent_path = bind(args.parent_directory / "report.json")
    if audit["input_bindings"].get(str(parent_path)) != inputs[str(parent_path)]:
        raise ValueError("pose diagnostic parent identity differs")
    parent = json.loads(parent_path.read_text())
    descriptor = parent.get("diagnostic_artifact", parent.get("diagnostic_output"))
    stored = load_arrays(bind(args.parent_directory / descriptor["path"], descriptor["sha256"]))
    rejected = audit["rejected_candidate"]
    poses = load_arrays(bind(audit_path.parent / rejected["path"], rejected["sha256"]))["qpos_native23"]
    frame = audit["after_control_self_contacts"]["worst_frame"] if args.frame is None else args.frame
    if type(frame) is not int or not 0 <= frame < len(poses):
        raise ValueError("pose diagnostic requires one valid retained frame")
    root = Path(__file__).resolve().parents[2]
    for path in (args.source_model, args.target_model, args.asset_root / MODEL, root / PHYSICS):
        resolved = bind(path)
        if audit["input_bindings"].get(str(resolved)) != inputs[str(resolved)]:
            raise ValueError("pose diagnostic cannot replace source/model/config")
    for path in collect_local_source_closure(root, [Path(__file__)]).files:
        bind(path)
    source_model = (
        mujoco.MjModel.from_binary_path(str(args.source_model))
        if args.source_model.suffix == ".mjb"
        else mujoco.MjModel.from_xml_path(str(args.source_model))
    )
    target_model = mujoco.MjModel.from_xml_path(str(args.target_model))
    _, physical_model, _ = prepare_true23_model(args.asset_root / MODEL, root / PHYSICS)
    compiled = {"source": compiled_model_sha256(source_model), "target": compiled_model_sha256(target_model)}
    physical_hash = compiled_model_sha256(physical_model)
    if compiled != audit["compiled_models"] or physical_hash != audit["compiled_physics_model_sha256"]:
        raise ValueError("pose diagnostic compiled model identities differ")
    selected = {
        key: (np.repeat(value[frame : frame + 1], 3, axis=0) if value.shape[0] == len(poses) else value.copy())
        for key, value in stored.items()
    }
    requested = selected["diagnostic_requested_qpos29"]
    candidate = {
        "joint_pos": requested[:, 7:],
        "root_pos_w": requested[:, :3],
        "root_quat_wxyz": requested[:, 3:7],
        "contact_flags": selected["diagnostic_contact_flags"],
        "fps": np.array([50.0]),
    }
    baseline = _restore_retained_diagnostic(
        source_model, target_model, candidate, selected, ik.RetargetConfig(**parent["ik_config"])
    )
    low, high = ik.safe_target_joint_bounds(target_model, native_action_clip=9.5, safe_limit_guard_rad=0.05)
    cfg = replace(OriginalTaskConfig(), maximum_joint_change_rad=float(np.max(high - low)))
    problem = OriginalTaskPath(
        source_model, target_model, requested, np.repeat(poses[frame : frame + 1, 7:], 3, axis=0), config=cfg
    )
    current = problem.serialized_variables(
        {
            "joint_pos": np.repeat(poses[frame : frame + 1, 7:], 3, axis=0),
            "body_pos_w": np.repeat(poses[frame : frame + 1, None, :3], 3, axis=0),
            "body_quat_w": np.repeat(poses[frame : frame + 1, None, 3:7], 3, axis=0),
        }
    )[0]
    fit = PoseEscape(
        problem,
        baseline,
        physical_model,
        upper_position_limit_m=args.upper_position_limit_m,
        interior_solver_margin=args.interior_solver_margin,
    )
    # Independently check the analytic local Jacobian before using it.
    analytic = fit.evaluate(current)[1]
    numerical = np.column_stack(
        [
            (fit.evaluate(current + np.eye(29)[j] * 1e-6)[0] - fit.evaluate(current - np.eye(29)[j] * 1e-6)[0])
            / 2e-6
            for j in range(29)
        ]
    )
    derivative_error = float(np.max(np.abs(analytic - numerical)))
    if derivative_error > 0.005:
        row, column = np.unravel_index(np.argmax(np.abs(analytic - numerical)), analytic.shape)
        raise ValueError(
            f"pose feasibility Jacobian failed finite-difference check: {derivative_error}; "
            f"row={row}, column={column}, analytic={analytic[row, column]}, numerical={numerical[row, column]}"
        )
    seeds = [("retained", current.copy())]
    neutral = current.copy()
    neutral[19:] = np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE)[13:]
    seeds.append(("neutral_upper", neutral))
    rng = np.random.default_rng(23)
    for i in range(6):
        seed = current.copy()
        # Only actual ten arm joints get alternative initial coordinates.
        seed[19:] = rng.uniform(low[13:], high[13:])
        seeds.append((f"fixed_random_upper_{i}", seed))
    lower, upper = problem.lower[0], problem.upper[0]
    metric = problem.posture_weight[:29]
    records, candidates = [], []
    for name, seed in seeds:
        answer = minimize(
            lambda x: 0.5 * np.dot(metric * (x - current), x - current),
            np.clip(seed, lower, upper),
            jac=lambda x: metric * (x - current),
            method="SLSQP",
            bounds=Bounds(lower, upper),
            constraints=[
                {"type": "ineq", "fun": lambda x: fit.evaluate(x)[0], "jac": lambda x: fit.evaluate(x)[1]}
            ],
            options={"maxiter": 160, "ftol": 1e-10, "disp": False},
        )
        values, _, residual, worst = fit.evaluate(answer.x, optimization=False)
        pose = problem.qpos(np.repeat(answer.x[None], 3, axis=0))[0]
        contacts = measure_self_contacts(physical_model, pose[None])
        protected = audit_norms(residual, fit.groups)
        accepted = (
            protected["passed"]
            and values.min() >= 0
            and contacts["frames_with_robot_robot_penetration"] == 0
            and np.all(answer.x >= lower)
            and np.all(answer.x <= upper)
        )
        row = {
            "seed": name,
            "optimizer_success": bool(answer.success),
            "optimizer_message": str(answer.message),
            "iterations": int(answer.nit),
            "pose_only_feasible": bool(accepted),
            "minimum_normalized_constraint": float(values.min()),
            "protected": protected,
            "contacts": contacts,
            "query_minimum_distance_m": worst["distance_m"] if worst else 0.03,
            "maximum_joint_change_from_retained_rad": float(np.max(np.abs(answer.x[6:] - current[6:]))),
        }
        records.append(row)
        candidates.append(pose)
        print(
            json.dumps(
                {
                    key: row[key]
                    for key in (
                        "seed",
                        "optimizer_success",
                        "iterations",
                        "pose_only_feasible",
                        "minimum_normalized_constraint",
                        "query_minimum_distance_m",
                    )
                }
            ),
            flush=True,
        )
    if compiled_model_sha256(physical_model) != physical_hash or any(
        sha256_file(Path(p)) != h for p, h in inputs.items()
    ):
        raise ValueError("pose diagnostic input/model changed during solve")
    output.mkdir(parents=True, exist_ok=False)
    archive = output / "pose_candidates.diagnostic-only.npz"
    with archive.open("xb") as stream:
        np.savez_compressed(
            stream, diagnostic_only_not_motion=np.array([True]), qpos_candidates=np.asarray(candidates)
        )
    report = {
        "kind": "g1_true23_multistart_single_pose_collision_diagnostic_v1",
        "frame": frame,
        "input_bindings": inputs,
        "compiled_models": compiled,
        "compiled_physics_model_sha256": physical_hash,
        "jacobian_max_abs_error": derivative_error,
        "results": records,
        "diagnostic_output": {"path": archive.name, "sha256": sha256_file(archive)},
        "full_path_temporal_constraints_tested": False,
        "input_pose_repeated_only_for_jacobian_interface": True,
        "all_original_per_frame_task_norms_and_safe_bounds_preserved": True,
        "separate_hand_head_pose_position_limit_m": args.upper_position_limit_m,
        "single_pose_limit_is_not_whole_clip_percentile_or_temporal_proof": True,
        "optimization_only_constraint_margin_fraction": fit.numerical_margin_fraction,
        "final_constraint_and_collision_acceptance_limits_unchanged": True,
        "physical_infeasibility_proven": False,
        "accepted_training_motion": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
    with (output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
