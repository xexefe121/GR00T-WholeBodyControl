"""Reproduce a rejected offline contact trial and compare predicted geometry.

This diagnostic intentionally stops at its first unchanged physical guard.
Prefix completion is not motion, policy, training-parent or hardware evidence.
"""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from gear_sonic.scripts.optimize_g1_true23_pd_trajectory import assert_bindings, load_arrays
from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_source_closure import collect_local_source_closure
from gear_sonic.utils.g1_true23_native_model_actuation import NativeModelActuationProfile
from gear_sonic.utils.g1_true23_pd_contact_step import contact_backward_pass
import gear_sonic.utils.g1_true23_pd_predictive_contact as predictive
from gear_sonic.utils.g1_true23_pd_protected_objective import ProtectedPdPlant
from gear_sonic.utils.g1_true23_pd_reactive_contact import make_reactive_contact_law
from gear_sonic.utils.g1_true23_pd_shooting import state_difference
from gear_sonic.utils.g1_true23_pd_source_history import verify_historical_report
from gear_sonic.utils.g1_true23_pd_trajectory_optimizer import canonical_target
from gear_sonic.utils.g1_true23_self_collision import SelfCollisionLinearizer, self_contact_rows
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-report", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--reproduction-report", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--regularization", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=0.125)
    parser.add_argument("--capture-from", type=int, default=380)
    parser.add_argument("--maximum-controls", type=int, default=500)
    parser.add_argument("--prediction-method", choices=("reactive", "predictive"), default="reactive")
    parser.add_argument("--planning-clearance-m", type=float, default=0.001)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    historical = verify_historical_report(
        args.trial_report, repository_root=root, source_archive=args.source_archive
    )
    report = json.loads(args.trial_report.read_text())
    reproduction = json.loads(args.reproduction_report.read_text())
    assert_bindings(reproduction["inputs"])
    cache = report["cached_linearization"]
    cache_path, nominal_path = Path(cache["path"]), Path(cache["nominal_trace_path"])
    seed_path = Path(reproduction["trace_path"])
    geometry_path = args.trial_report.parent / "contact_constraints.npz"
    paths = [
        args.trial_report,
        args.reproduction_report,
        args.source_archive,
        cache_path,
        nominal_path,
        seed_path,
        geometry_path,
        args.asset_root / MODEL,
        root / PHYSICS,
        *collect_local_source_closure(root, [Path(__file__)]).files,
    ]
    bindings = {str(path.resolve()): sha256_file(path) for path in paths}
    assert_bindings(
        {
            str(cache_path): cache["sha256"],
            str(nominal_path): cache["nominal_trace_sha256"],
            str(seed_path): reproduction["trace_sha256"],
            str(geometry_path.resolve()): report["inputs"][str(geometry_path.resolve())],
        }
    )
    _, model, _ = prepare_true23_model(args.asset_root / MODEL, root / PHYSICS)
    captures, current_index = [], -1

    class DiagnosticPlant(ProtectedPdPlant):
        def _tick(self, data, target):
            try:
                return super()._tick(data, target)
            finally:
                if current_index >= args.capture_from:
                    captures.append(
                        dict(
                            control=current_index,
                            physics_step=self.physics_steps,
                            contact_geometry_from_pre_step=[
                                dict(geoms=row["geoms"], distance_m=row["distance_m"])
                                for row in self_contact_rows(model, data)
                            ],
                        )
                    )

    plant = DiagnosticPlant(model, NativeModelActuationProfile.from_sim_config(root / PHYSICS))
    if plant.model_sha != report["compiled_physics_sha256"]:
        raise ValueError("contact prediction diagnostic changed original physics")
    limits = {
        tuple(row["geoms"]): row["distance_m"]
        for row in report["initial_research_self_contact_envelope"]["minimum_self_distance_by_pair"]
    }
    plant.set_research_contact_envelope(limits)
    seed, nominal, local, packed = [
        load_arrays(path) for path in (seed_path, nominal_path, cache_path, geometry_path)
    ]
    controls = nominal["target23"]
    geometry = []
    for index in range(len(controls)):
        mask = packed["control_index"] == index
        geometry.append({key: packed[key][mask] for key in ("jacobian", "distance", "floor", "geoms")})
    stages = []
    contact_backward_pass(
        plant, local, controls, seed["target23"], geometry, args.regularization, stage_models=stages
    )
    last_prediction = {}

    class TracedPredictor(predictive.SubstepContactPredictor):
        def poses(self, state, target):
            poses, velocities = super().poses(state, target)
            self.latest = dict(
                integration_state=np.asarray(state).tolist(),
                target23=np.asarray(target).tolist(),
                qpos=poses.tolist(),
                qvel=velocities.tolist(),
            )
            self.latest_velocities = velocities
            return poses, velocities

        def rows(self, poses, limits, clearance_m):
            rows = super().rows(poses, limits, clearance_m)
            last_prediction.clear()
            last_prediction.update(self.latest)
            last_prediction["control"] = current_index
            last_prediction["contacts"] = [
                dict(
                    substep=row["substep"],
                    geoms=row["geoms"],
                    distance_m=row["distance"],
                    planning_floor_m=row["floor"],
                    normal_speed_m_s=float(row["jacobian"] @ self.latest_velocities[row["substep"]]),
                )
                for row in rows
            ]
            return rows

    if args.prediction_method == "predictive":
        # This patches only this diagnostic process's private factory. The
        # tested implementation and every external simulator process are unchanged.
        with patch.object(predictive, "SubstepContactPredictor", TracedPredictor):
            law, stats = predictive.make_predictive_contact_law(
                plant, controls, stages, args.alpha, limits, clearance_m=args.planning_clearance_m
            )
    else:
        law, stats = make_reactive_contact_law(
            plant, local, nominal, controls, stages, args.alpha, limits, clearance_m=args.planning_clearance_m
        )
    query = SelfCollisionLinearizer(model, near_distance_m=0.03)

    def distances(pose):
        return [
            dict(geoms=row["geoms"], distance_m=row["distance_m"]) for row in query.pose_rows(pose, np.arange(29))
        ]

    data = plant.scratch(seed["integration_state"][0])
    previous, predictions, failure = np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE), [], None
    for current_index in range(min(args.maximum_controls, len(controls))):
        index = current_index
        nominal_previous = controls[index - 1] if index else np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE)
        difference = np.r_[
            state_difference(model, nominal["qpos"][index], nominal["qvel"][index], data.qpos, data.qvel),
            previous - nominal_previous,
        ]
        try:
            delta = (
                law(index, difference, plant.state(data))
                if args.prediction_method == "predictive"
                else law(index, difference)
            )
            target = canonical_target(np.clip(controls[index] + delta, plant.lower, plant.upper))
            if index >= args.capture_from:
                base_pose, proposed_pose = nominal["qpos"][index + 1].copy(), nominal["qpos"][index + 1].copy()
                shift = local["a"][index] @ difference
                mujoco.mj_integratePos(model, base_pose, shift[:29], 1)
                mujoco.mj_integratePos(
                    model, proposed_pose, (shift + local["b"][index] @ (target - controls[index]))[:29], 1
                )
                prediction = dict(
                    control=index,
                    current_geometry=distances(data.qpos),
                    predicted_without_control_update=distances(base_pose),
                    predicted_with_emitted_control=distances(proposed_pose),
                    state_difference_max=float(np.max(np.abs(difference[:58]))),
                    target_update_max_rad=float(np.max(np.abs(target - controls[index]))),
                )
                predictions.append(prediction)
            plant.integrate_control(data, target)
            if index >= args.capture_from:
                prediction["actual_next_geometry"] = distances(data.qpos)
            previous = target
        except (ValueError, RuntimeError) as exc:
            failure = dict(control=index, physics_step=plant.physics_steps, error=str(exc))
            break
    assert_bindings(bindings)
    plant.assert_unchanged()
    result = dict(
        kind="g1_true23_rejected_contact_prediction_diagnostic_v1",
        inputs=bindings,
        historical_input_evidence=historical,
        regularization=args.regularization,
        alpha=args.alpha,
        failure=failure,
        predictions=predictions,
        physics_contacts=captures,
        reactive_stats=stats,
        prediction_method=args.prediction_method,
        planning_clearance_m=args.planning_clearance_m,
        last_actual_state_prediction=last_prediction,
        original_compiled_physics_sha256=plant.model_sha,
        prefix_diagnostic_not_complete_motion_evidence=True,
        admissible_training_parent=False,
        simulator_qualified=False,
        deployment_ready=False,
        hardware_authorized=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(args.output), failure=failure, reactive_stats=stats)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
