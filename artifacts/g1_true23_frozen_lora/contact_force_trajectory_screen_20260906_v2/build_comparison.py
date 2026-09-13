"""Recheck complete force-restoration evidence without training or hardware I/O."""

import json
from pathlib import Path
import runpy

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "artifacts/g1_true23_frozen_lora"
REFINED = ARTIFACTS / "contact_force_trajectory_20260906_v2"
ENVELOPE = ARTIFACTS / "contact_force_trajectory_envelope_20260906_v2"
PREVIOUS = ARTIFACTS / "contact_trajectory_screen_20260906_v1"
OUTPUT = Path(__file__).resolve().parent / "comparison.json"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    # Reuse the previously validated, read-only hash and full-suite audits;
    # bind their source plus every original evidence identity again below.
    prior_helpers = runpy.run_path(str(PREVIOUS / "build_comparison.py"))
    bind, read, bind_map = (prior_helpers[name] for name in ("bind", "read", "bind_map"))
    identities = prior_helpers["IDENTITIES"]
    expected = prior_helpers["EXPECTED"]
    bind(Path(__file__))
    bind(PREVIOUS / "build_comparison.py")
    force_v1 = read(ARTIFACTS / "force_trajectory_screen_20260906_v1/comparison.json")
    bind_map(force_v1["files"])
    previous = read(PREVIOUS / "comparison.json")
    bind_map(previous["files"])
    assert previous["same_policy_and_actuation_contract_verified"] is True
    assert previous["clips"] == 8 and previous["frames"] == 6035
    refinement = read(REFINED / "report.json")
    manifest = read(REFINED / "motions.json")
    bind_map(refinement["inputs"])
    assert refinement["kind"] == "g1_true23_complete_contact_force_refinement_v2"
    assert refinement["clips_in"] == refinement["clips_out"] == len(expected)
    assert [row["name"] for row in refinement["records"]] == list(expected)
    assert [row["name"] for row in manifest["motions"]] == list(expected)
    assert refinement["torque_limit_multiplier"] == 0.2375
    assert refinement["optimizer_effort_fraction"] == 0.98
    assert refinement["optimizer"]["patch_guard_m"] == 0.00005
    assert refinement["versions"]["clarabel"] == "0.11.1"
    assert refinement["production_training_changed"] is False
    assert refinement["dynamic_feasibility_proven"] is False
    for report in (refinement, manifest):
        for key in ("teacher_accepted", "hardware_authorized", "deployment_ready"):
            assert report[key] is False
    old_suite, old_cases = prior_helpers["inspect_suite"](
        ARTIFACTS / "contact_trajectory_envelope_20260906_v1/suite_summary.json"
    )
    byte_identical_references = all(
        row["output_sha256"] == row["previous_candidate_sha256"] for row in refinement["records"]
    )
    if byte_identical_references:
        # No new physics run is claimed when every reference byte is unchanged.
        # Existing full requested failures are rebound to identical inputs.
        new_suite, new_cases = old_suite, old_cases
    else:
        new_suite, new_cases = prior_helpers["inspect_suite"](ENVELOPE / "suite_summary.json")
    assert old_suite["diagnostic_pair"] == new_suite["diagnostic_pair"] == previous["diagnostic_pair"]
    contacts, force_balance, replays = [], [], []
    previous_contacts = {row["name"]: row for row in previous["contacts"]}
    previous_forces = {row["name"]: row for row in previous["force_balance"]}
    previous_support = read(ARTIFACTS / "contact_trajectory_support_20260906_v1/report.json")
    previous_support_models = {
        row["name"]: {
            name: {k: value[k] for k in ("torque_limits_nm", "compiled_mjb_sha256")}
            for name, value in row["models"].items()
        }
        for row in previous_support["records"]
    }
    del previous_support
    for entry, row in zip(manifest["motions"], refinement["records"]):
        name, frames = row["name"], expected[row["name"]]
        assert row == read(REFINED / f"{name}.report.json")
        assert entry["path"] == row["output"] and entry["weight"] == row["weight"]
        assert row["frames_in"] == row["frames_out"] == frames and row["frames_dropped"] == 0
        assert row["controlled_joint_count"] == 23 and row["time_scale"] == 1.0
        assert row["root_orientation_changed"] is False
        assert row["root_orientation_serialization_error_rad"] <= 2e-7
        assert row["serialized_temporal_audit"]["passed"]
        assert row["serialized_fk_audit"]["position_fk_consistent"]
        assert row["serialized_fk_audit"]["orientation_fk_consistent"]
        assert row["causal_terms"]["packets_rebuilt_and_validated"] == frames - 10
        assert row["causal_terms"]["stored_or_sent_to_robot"] is False
        for key in ("teacher_accepted", "hardware_authorized", "deployment_ready", "dynamic_feasibility_proven"):
            assert row[key] is False and row["refinement"][key] is False
        assert row["refinement"]["pose_variables"] == frames * 26
        assert row["refinement"]["all_frames_have_independent_pose_variables"] is True
        assert row["refinement"]["pose_objective_center"] == "current_iterate"
        for iteration in row["refinement"]["iterations"]:
            for attempt in iteration["attempts"]:
                qp = attempt["qp"]
                assert qp["path_variables"] == frames * 26
                assert qp["all_frames_have_independent_pose_variables"] is True
                assert qp["pose_objective_center"] == "current_iterate"
                assert qp["backend"] == "clarabel" and qp["backend_version"] == "0.11.1"
                if qp["accepted"]:
                    assert qp["status"] == "Solved"
                    assert qp["independent_maximum_scaled_linear_violation"] <= 1e-8
                for trial in attempt["trials"]:
                    if trial["accepted"]:
                        assert qp["accepted"] and trial["temporal_passed"]
                        assert trial["frozen_patch_violation_m"] <= 2e-7
        bind(row["output"], row["output_sha256"])
        with np.load(row["output"], allow_pickle=False) as archive:
            assert archive["joint_pos"].shape == (frames, 23)
            assert all(np.isfinite(archive[key]).all() for key in archive.files)
        prior_record = read(ARTIFACTS / f"contact_trajectory_20260906_v2/{name}.report.json")
        assert row["previous_candidate_sha256"] == prior_record["output_sha256"]
        assert row["original_source_sha256"] == prior_record["source_sha256"]
        before, after = row["refinement"]["before"]["contact"], row["serialized_contact_audit"]
        assert before == prior_record["serialized_contact_audit"]
        contacts.append(
            {
                "name": name,
                "frames": frames,
                "previous": previous_contacts[name],
                "refined": {
                    "violated_frames": after["violated_frames"],
                    "maximum_violation_m": after["maximum_violation_m"],
                    "passed": after["passed"],
                    "worst": after["worst"][:2],
                    "geometry": {
                        model: {
                            k: v
                            for k, v in value.items()
                            if k in ("frames", "frames_with_floor_overlap", "worst_floor_overlap_m")
                        }
                        for model, value in row["geometry_after"].items()
                    },
                },
                "failure": row["refinement"]["failure"],
                "iterations": [
                    {k: v for k, v in item.items() if k in ("iteration", "attempts", "fraction")}
                    for item in row["refinement"]["iterations"]
                ],
            }
        )
        support_identity = row["independent_support_report"]
        bind(support_identity["path"], support_identity["sha256"])
        support = read(support_identity["path"])
        assert support["name"] == name and support["output_sha256"] == row["output_sha256"]
        force_row = {"name": name, "models": {}}
        for model, digest in refinement["compiled_models"].items():
            value = support["models"][model]
            assert row["independent_support"][model] == {k: v for k, v in value.items() if k != "rows"}
            assert value["compiled_mjb_sha256"] == digest
            assert value["frames_checked"] == len(value["rows"]) == frames
            assert value["frames_dropped"] == 0
            assert value["mode"] == "reference_inverse_dynamics"
            assert value["candidate_floor_gap_tolerance_m"] == 0.002
            assert not value["velocity_and_acceleration_assumed_zero"]
            assert not value["dynamic_feasibility_proven"]
            assert not value["contact_complementarity_or_support_kinematics_proven"]
            assert not value["hardware_authorized"] and not value["deployment_ready"]
            prior_value = previous_support_models[name][model]
            assert value["torque_limits_nm"] == prior_value["torque_limits_nm"]
            assert value["compiled_mjb_sha256"] == prior_value["compiled_mjb_sha256"]
            expected_headroom = sum(
                item["within_supplied_effort_limits"]
                and item["minimum_peak_effort_ratio"] <= 0.98 + 1e-8
                for item in value["rows"]
            )
            assert row["frames_with_requested_effort_headroom"][model] == expected_headroom
            force_row["models"][model] = {
                "frames_with_requested_2_percent_algebraic_headroom": expected_headroom,
                "stance": previous_forces[name]["models"][model]["previous"],
                "contact_only": previous_forces[name]["models"][model]["refined"],
                "force_restored": {k: v for k, v in value.items() if k.startswith("frames_")},
            }
        force_balance.append(force_row)
        assert row["all_frames_with_conditional_force_solution"] == all(
            item["frames_with_conditional_solution_within_effort_limits"] == frames
            for item in support["models"].values()
        )
    contract_keys = (
        "gain_profile",
        "gain_kp_hardware",
        "gain_kd_hardware",
        "effort_limit_hardware_nm",
        "action_fraction",
        "ankle_effort_nm",
        "target_slew_rad_s",
        "initial_state",
        "observation_timing",
        "body_tracking_timing",
        "previous_action_semantics",
        "stateful_native_controller",
        "active_effort_target_projection",
    )
    for key in new_cases:
        old, new = old_cases[key], new_cases[key]
        assert all(old[field] == new[field] for field in contract_keys)
        replays.append(
            {
                "name": key[0],
                "measured_start": key[1],
                "contact_only": prior_helpers["compact_case"](old),
                "force_restored": prior_helpers["compact_case"](new),
            }
        )
    witness = read(
        ARTIFACTS / "force_trajectory_minimum_step_static_crouch_20260906_v1/serialized_witness_audit.json"
    )
    bind_map(witness["files"])
    assert witness["single_pose_has_conditional_force_solution_in_both_models"] is True
    assert witness["full_clip_test"] is False and witness["deployment_ready"] is False
    assert witness["compiled_models"] == refinement["compiled_models"]
    assert witness["torque_limit_multiplier"] == refinement["torque_limit_multiplier"]
    assert all(prior_helpers["sha256"](Path(path)) == digest for path, digest in identities.items())
    result = {
        "kind": "g1_true23_contact_force_refinement_comparison_v2",
        "clips": len(expected),
        "frames": sum(expected.values()),
        "causal_packets_rebuilt_and_validated": sum(expected.values()) - 10 * len(expected),
        "optimizer": refinement["optimizer"],
        "contacts": contacts,
        "force_balance": force_balance,
        "replays": replays,
        "diagnostic_pair": new_suite["diagnostic_pair"],
        "runtime_versions": refinement["versions"],
        "separate_stationary_witness": {
            "source_frame": witness["source_frame"],
            "full_clip_test": False,
            "robust_effort_headroom_proven": False,
            "models": {
                name: {
                    "conditional_force_frames": value["frames_with_conditional_solution_within_effort_limits"],
                    "maximum_finite_minimum_effort_ratio": value["maximum_finite_minimum_effort_ratio"],
                }
                for name, value in witness["support"].items()
            },
        },
        "independently_rechecked_file_count": len(identities),
        "files": identities,
        "same_policy_and_actuation_contract_verified": True,
        "all_eight_reference_files_byte_identical_to_previous": byte_identical_references,
        "new_closed_loop_replays_run": not byte_identical_references,
        "previous_full_requested_replay_evidence_reused_on_identical_reference_bytes": byte_identical_references,
        "original_full_weight_v14_matched_budget_comparison": False,
        "training_or_controller_changed": False,
        "teacher_accepted": False,
        "dynamic_feasibility_proven": False,
        "hardware_authorized": False,
        "deployment_ready": False,
        "limitations": [
            "All original supplied frames, 23 joints, timing and root orientation remain represented.",
            "Reference force optimization is offline, not a live PICO retargeter or robot controller.",
            "The full-force LP uses conditional contacts and optimistic friction assistance, not no-slip proof.",
            "This fixed-policy comparison does not establish retrained tracking or exact 29-DoF parity.",
            "Recorded-posture simulation does not establish a physical Unitree FSM transfer or motor-fault cause.",
        ],
    }
    with OUTPUT.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"output": str(OUTPUT), "sha256": prior_helpers["sha256"](OUTPUT), "files": len(identities)}))


if __name__ == "__main__":
    main()
