"""Independently replay a complete offline PD candidate; never qualify hardware."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.scripts.optimize_g1_true23_pd_trajectory import (
    assert_bindings,
    evaluate_trajectory,
    load_arrays,
)
from gear_sonic.utils.g1_true23_generalist_benchmark import LANDMARKS, MODEL, PHYSICS
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_source_closure import collect_local_source_closure
from gear_sonic.utils.g1_true23_native_model_actuation import NativeModelActuationProfile
from gear_sonic.utils.g1_true23_pd_protected_objective import ProtectedMotionObjective
from gear_sonic.utils.g1_true23_pd_shooting import PdShootingPlant
from gear_sonic.utils.g1_true23_pd_source_history import verify_historical_report
from gear_sonic.utils.g1_true23_pd_trajectory_optimizer import MotionObjective
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model


class ContactObservedPlant(PdShootingPlant):
    """Read actual solved contacts without changing integration or collision masks."""

    def __init__(self, model, profile):
        super().__init__(model, profile)
        self.floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.feet = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ("left_ankle_roll_link", "right_ankle_roll_link")
        ]
        if self.floor < 0 or min(self.feet) < 0:
            raise ValueError("contact audit requires original named floor and native feet")
        self.contact_rows, self.loaded_pairs = [], {}
        self.all_self_distances, self.nonfoot_penetration_samples = {}, 0
        self.terminal_geometry_checked = False

    def observe_geometry(self, data):
        for contact in data.contact[: data.ncon]:
            if contact.dist >= 0:
                continue
            geoms = tuple(sorted((int(contact.geom1), int(contact.geom2))))
            bodies = [int(self.model.geom_bodyid[g]) for g in geoms]
            if 0 in bodies:
                if bodies[1 - bodies.index(0)] not in self.feet:
                    self.nonfoot_penetration_samples += 1
            else:
                self.all_self_distances[geoms] = min(self.all_self_distances.get(geoms, 0.0), float(contact.dist))

    def observe_terminal_geometry(self, qpos, qvel):
        probe = mujoco.MjData(self.model)
        probe.qpos[:], probe.qvel[:] = qpos, qvel
        mujoco.mj_forward(self.model, probe)
        self.observe_geometry(probe)  # Geometry only; these probe forces are never reported as actual forces.
        self.terminal_geometry_checked = True

    def _tick(self, data, target):
        result = super()._tick(data, target)
        self.observe_geometry(data)
        # mj_step's contacts/constraint solution describe the just-integrated
        # physical step. mj_contactForce only reads that solution.
        # https://mujoco.readthedocs.io/en/3.5.0/APIreference/APIfunctions.html#mj-contactforce
        foot_force, nonfoot_ground, self_contact, penetration = np.zeros(2), 0, 0, 0.0
        for i in range(data.ncon):
            contact, wrench = data.contact[i], np.zeros(6)
            mujoco.mj_contactForce(self.model, data, i, wrench)
            if not np.isfinite(wrench).all():
                raise ValueError("contact audit found a nonfinite solved force")
            if wrench[0] <= 1e-6:
                continue
            geoms = (int(contact.geom1), int(contact.geom2))
            bodies = tuple(int(self.model.geom_bodyid[g]) for g in geoms)
            pair = tuple(sorted(geoms))
            item = self.loaded_pairs.setdefault(
                pair,
                dict(
                    geom_ids=list(pair),
                    body_names=[
                        mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[g]))
                        for g in pair
                    ],
                    loaded_contact_samples=0,
                    normal_force_max_n=0.0,
                ),
            )
            item["loaded_contact_samples"] += 1
            item["normal_force_max_n"] = max(item["normal_force_max_n"], float(wrench[0]))
            penetration = max(penetration, -float(contact.dist))
            if self.floor in geoms:
                robot_body = bodies[1 - geoms.index(self.floor)]
                if robot_body in self.feet:
                    foot_force[self.feet.index(robot_body)] += wrench[0]
                else:
                    nonfoot_ground += 1
            elif bodies[0] != 0 and bodies[1] != 0:
                self_contact += 1
        self.contact_rows.append([*foot_force, nonfoot_ground, self_contact, penetration])
        return result

    def contact_evidence(self):
        rows = np.asarray(self.contact_rows)
        return dict(
            physics_substeps_observed=len(rows),
            loaded_reporting_threshold_n=1e-6,
            nonfoot_ground_loaded_substeps=int(np.count_nonzero(rows[:, 2])),
            robot_self_contact_loaded_substeps=int(np.count_nonzero(rows[:, 3])),
            foot_normal_force_max_n=np.max(rows[:, :2], axis=0).tolist(),
            loaded_contact_penetration_max_m=float(np.max(rows[:, 4])),
            observed_loaded_pairs=list(self.loaded_pairs.values()),
            all_penetrating_self_pairs=[
                dict(geoms=list(pair), distance_m=distance)
                for pair, distance in sorted(self.all_self_distances.items())
            ],
            nonfoot_ground_penetration_samples=self.nonfoot_penetration_samples,
            terminal_geometry_checked=self.terminal_geometry_checked,
            collision_model_coverage_not_qualified=True,
            contact_timing_or_slip_qualified=False,
        )


def verify_protected_nonregression(initial_metrics, candidate_metrics, initial_contacts, candidate_contacts):
    initial = initial_metrics["lifecycle"]["source_motion_tracking"]
    candidate = candidate_metrics["lifecycle"]["source_motion_tracking"]
    if not candidate["full_source_motion_completed"]:
        raise ValueError("protected verification requires complete source playback")
    for (name, _, _), threshold in zip(LANDMARKS, (0.05, 0.05, 0.10, 0.10, 0.10), strict=True):
        ceiling = max(initial["landmark_position_p95_m"][name], threshold)
        value = candidate["landmark_position_p95_m"][name]
        if not np.isfinite(value) or value > ceiling + 1e-12:
            raise ValueError("independent protected source-landmark verification failed: " + name)
    if candidate_contacts["nonfoot_ground_penetration_samples"]:
        raise ValueError("independent protected verification found non-foot ground penetration")
    limits = {tuple(row["geoms"]): row["distance_m"] for row in initial_contacts["all_penetrating_self_pairs"]}
    for row in candidate_contacts["all_penetrating_self_pairs"]:
        if row["distance_m"] < limits.get(tuple(row["geoms"]), 0.0) - 1e-6:
            raise ValueError("independent protected self-contact verification failed: " + str(row["geoms"]))
    if not candidate_contacts["terminal_geometry_checked"]:
        raise ValueError("protected verification must include final-state geometry")
    return dict(
        original_landmark_ceiling_nonregression_verified=True,
        actual_self_contact_envelope_nonregression_verified=True,
        final_state_contact_geometry_checked=True,
        these_research_envelopes_do_not_qualify_collision_or_hardware_safety=True,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer-report", type=Path, required=True)
    parser.add_argument("--reproduction-report", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    report = json.loads(args.optimizer_report.read_text())
    reproduction = json.loads(args.reproduction_report.read_text())
    historical = verify_historical_report(
        args.optimizer_report, repository_root=root, source_archive=args.source_archive
    )
    assert_bindings(reproduction["inputs"])
    if sha256_file(args.reproduction_report) != report["reproduction_report_sha256"]:
        raise ValueError("candidate uses a different seed reproduction")
    trace_path, seed_path = Path(report["candidate_trace_path"]), Path(reproduction["trace_path"])
    if sha256_file(trace_path) != report["candidate_trace_sha256"]:
        raise ValueError("candidate trace changed after optimization")
    if sha256_file(seed_path) != reproduction["trace_sha256"]:
        raise ValueError("candidate seed changed")
    parent_path = Path(reproduction["parent_evaluation"])
    parent = json.loads(parent_path.read_text())
    reference_path = Path(report["original_requested_motion_path"])
    if sha256_file(reference_path) != report["original_requested_motion_sha256"]:
        raise ValueError("candidate reference changed")
    _, model, _ = prepare_true23_model(args.asset_root / MODEL, root / PHYSICS)
    plant = ContactObservedPlant(model, NativeModelActuationProfile.from_sim_config(root / PHYSICS))
    if plant.model_sha != report["compiled_physics_sha256"]:
        raise ValueError("candidate verification changed original physics")
    paths = [
        args.optimizer_report,
        args.reproduction_report,
        trace_path,
        seed_path,
        parent_path,
        reference_path,
        args.asset_root / MODEL,
        root / PHYSICS,
        *collect_local_source_closure(root, [Path(__file__)]).files,
    ]
    if historical["archive_path"] is not None:
        paths.append(Path(historical["archive_path"]))
    paths.extend(Path(row["archived_path"]) for row in historical["archived_python_sources_used"])
    bindings = {str(path.resolve()): sha256_file(path) for path in paths}
    candidate, seed = load_arrays(trace_path), load_arrays(seed_path)
    np.testing.assert_array_equal(candidate["integration_state"][0], seed["integration_state"][0])
    if len(candidate["target23"]) != report["complete_controls"]:
        raise ValueError("candidate omits controls")
    replay = plant.rollout(seed["integration_state"][0], candidate["target23"], physics_records=True)
    for key, value in replay.items():
        if value.dtype != candidate[key].dtype or value.shape != candidate[key].shape:
            raise ValueError("candidate physical array contract changed: " + key)
        if value.tobytes() != candidate[key].tobytes():
            raise ValueError("candidate physical replay is not bit exact: " + key)
    objective_kind = report["objective_contract"].get("kind", "original_squared_motion_v1")
    if objective_kind == "g1_true23_original_motion_per_landmark_and_self_clearance_v1":
        objective_class = ProtectedMotionObjective
    elif objective_kind == "original_squared_motion_v1":
        objective_class = MotionObjective
    else:
        raise ValueError("verification requires a known exact objective contract")
    objective = objective_class(plant, load_arrays(reference_path), parent["timeline"])
    metrics, errors = evaluate_trajectory(plant, objective, parent["timeline"], replay)
    if "landmark_error_m" in candidate:
        np.testing.assert_array_equal(errors, candidate["landmark_error_m"])
    cost = objective.cost(replay, candidate["target23"], seed["target23"])
    if cost != report["final_cost"]:
        raise ValueError("candidate full-reference objective did not reproduce exactly")
    plant.observe_terminal_geometry(replay["qpos"][-1], replay["qvel"][-1])
    protection = None
    if objective_class is ProtectedMotionObjective:
        previous = report.get("historical_initialization_evidence")
        initial_trace_path = seed_path
        if previous is not None:
            initial_parent_path = Path(previous["historical_report_path"]).resolve(strict=True)
            if sha256_file(initial_parent_path) != report["inputs"].get(str(initial_parent_path)):
                raise ValueError("protected initialization report is not an unchanged optimizer input")
            bindings[str(initial_parent_path)] = report["inputs"][str(initial_parent_path)]
            initial_parent = json.loads(initial_parent_path.read_text())
            initial_trace_path = Path(initial_parent["candidate_trace_path"]).resolve(strict=True)
            paths.append(initial_parent_path)
        if sha256_file(initial_trace_path) != report["inputs"].get(str(initial_trace_path)):
            raise ValueError("protected initialization trajectory is not an unchanged optimizer input")
        bindings[str(initial_trace_path)] = report["inputs"][str(initial_trace_path)]
        paths.append(initial_trace_path)
        initial_saved = load_arrays(initial_trace_path)
        initial_plant = ContactObservedPlant(model, plant.profile)
        initial_replay = initial_plant.rollout(
            seed["integration_state"][0], initial_saved["target23"], physics_records=True
        )
        for key, value in initial_replay.items():
            if (
                value.shape != initial_saved[key].shape
                or value.dtype != initial_saved[key].dtype
                or value.tobytes() != initial_saved[key].tobytes()
            ):
                raise ValueError("protected initialization independent physical replay failed: " + key)
        initial_contacts = initial_plant.contact_evidence()
        actual_initial_envelope = initial_contacts["all_penetrating_self_pairs"]
        if (
            actual_initial_envelope
            != report["initial_research_self_contact_envelope"]["minimum_self_distance_by_pair"]
        ):
            raise ValueError("protected initial self-contact envelope did not reproduce independently")
        initial_plant.observe_terminal_geometry(initial_replay["qpos"][-1], initial_replay["qvel"][-1])
        initial_metrics, _ = evaluate_trajectory(initial_plant, objective, parent["timeline"], initial_replay)
        verify_protected_nonregression(
            initial_metrics, initial_metrics, initial_contacts, initial_plant.contact_evidence()
        )
        protection = verify_protected_nonregression(
            initial_metrics, metrics, initial_contacts, plant.contact_evidence()
        )
        if report.get("terminal_nonregression_required", False):
            for field in (
                "final_proof_standing_joint_error_max_rad",
                "final_proof_root_speed_max_m_s",
                "final_proof_root_position_error_max_m",
                "final_proof_root_orientation_error_max_rad",
            ):
                original, value = initial_metrics["lifecycle"][field], metrics["lifecycle"][field]
                if (
                    original is None
                    or value is None
                    or not np.isfinite([original, value]).all()
                    or value > original + 1e-12
                ):
                    raise ValueError("independent terminal nonregression failed: " + field)
            protection["all_four_terminal_metrics_nonregression_verified"] = True
        for path in paths:
            bindings.setdefault(str(path.resolve()), sha256_file(path))
    assert_bindings(bindings)
    args.output_directory.mkdir(parents=True, exist_ok=False)
    evidence = dict(
        kind="g1_true23_offline_pd_candidate_exact_replay_verification_v1",
        inputs=bindings,
        complete_controls=objective.count,
        physical_and_integration_state_arrays_bit_exact=sorted(replay),
        original_reference_metrics_reproduced=metrics,
        full_objective_reproduced_exactly=cost,
        original_compiled_physics_sha256=plant.model_sha,
        actual_contact_observations=plant.contact_evidence(),
        historical_implementation_evidence=historical,
        independent_protection_verification=protection,
        state_writes_after_initial_condition=0,
        future_motion_used_offline=True,
        not_a_sonic_policy_or_causal_teleop_controller=True,
        admissible_training_parent=False,
        simulator_qualified=False,
        hardware_authorized=False,
        deployment_ready=False,
    )
    with (args.output_directory / "report.json").open("x") as stream:
        json.dump(evidence, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                key: evidence[key]
                for key in (
                    "complete_controls",
                    "physical_and_integration_state_arrays_bit_exact",
                    "full_objective_reproduced_exactly",
                    "deployment_ready",
                )
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
