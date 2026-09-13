"""Recheck complete contact-refinement evidence; never train or command hardware."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "artifacts/g1_true23_frozen_lora"
REFINED = ARTIFACTS / "contact_trajectory_20260906_v2"
SUPPORT = ARTIFACTS / "contact_trajectory_support_20260906_v1"
ENVELOPE = ARTIFACTS / "contact_trajectory_envelope_20260906_v1"
PREVIOUS = ARTIFACTS / "stance_retarget_20260906_v2"
BASELINE = ARTIFACTS / "projection_cost_20260906_v1/baseline100"
OUTPUT = Path(__file__).resolve().parent / "comparison.json"
EXPECTED = {
    "pico_upright_anchor": 1024,
    "pico_standing_anchor": 1024,
    "pico_crouch_anchor": 1024,
    "pico_twist2_walk_001": 695,
    "pico_twist2_walk_010": 510,
    "original_sonic_hand_crawl": 606,
    "original_sonic_elbow_crawl": 606,
    "original_sonic_happy_dance": 546,
}
IDENTITIES = {}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bind(path, expected=None):
    path = Path(path).resolve(strict=True)
    if not any(path.is_relative_to(root) for root in (ROOT, ROOT.parent / "GR00T-WholeBodyControl")):
        raise ValueError(f"evidence outside the explicit workspace/source roots: {path}")
    digest = sha256(path)
    if (expected is not None and digest != expected) or IDENTITIES.get(str(path), digest) != digest:
        raise ValueError(f"changed evidence: {path}")
    IDENTITIES[str(path)] = digest
    return path


def read(path):
    return json.loads(bind(path).read_text())


def bind_map(values):
    for path, digest in values.items():
        bind(path, digest)


def bind_list(values):
    for value in values:
        bind(value["path"], value["sha256"])


def inspect_suite(path):
    suite = read(path)
    assert suite["reference_clip_count"] == 8 and suite["reference_frame_count"] == 6035
    assert suite["case_count"] == 9
    assert suite["deployment_ready"] is False and suite["hardware_authorized"] is False
    bind_list(suite["inputs"])
    cases = {}
    for row in suite["cases"]:
        key = (row["name"], row["measured_start"])
        assert key not in cases
        bind_list([row["summary"], *row["trajectories"]])
        summary = read(row["summary"]["path"])
        bind_map(summary["sources"])
        assert summary["cases"] == [row["result"]]
        assert summary["diagnostic_pair"] == suite["diagnostic_pair"]
        assert row["result"]["requested_transitions"] == EXPECTED[row["name"]] - 11
        cases[key] = row["result"]
    assert set(cases) == {(name, False) for name in EXPECTED} | {("original_sonic_happy_dance", True)}
    return suite, cases


def compact_case(row):
    result = {
        key: row[key]
        for key in (
            "requested_transitions",
            "completed_transitions",
            "completed_active_physics_steps",
            "failure",
            "maximums",
            "motion_fidelity",
            "paired_lifecycle_simulator_screen_passed",
        )
    }
    for phase in ("startup_hold", "return_hold"):
        if row["initial_state"] == "measured":
            result[phase] = {
                key: row[phase][key]
                for key in (
                    "requested_transitions",
                    "completed_transitions",
                    "completed_physics_steps",
                    "standing_screen_passed",
                    "failure",
                    "failure_details",
                )
            }
    return result


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    bind(Path(__file__))
    refinement = read(REFINED / "report.json")
    manifest = read(REFINED / "motions.json")
    support = read(SUPPORT / "report.json")
    previous_support = read(PREVIOUS / "support_dynamics/report.json")
    bind_map(refinement["inputs"])
    for report in (support, previous_support):
        assert report["mode"] == "reference_inverse_dynamics"
        assert report["torque_limit_multiplier"] == 0.2375
        assert report["deployment_ready"] is False and report["dynamic_feasibility_proven"] is False
        bind_map(report["inputs"])
        bind_map(report["sources"])
    old_suite, old_cases = inspect_suite(BASELINE / "envelope_stance_v2/suite_summary.json")
    new_suite, new_cases = inspect_suite(ENVELOPE / "suite_summary.json")
    assert old_suite["diagnostic_pair"] == new_suite["diagnostic_pair"]
    assert refinement["clips_in"] == refinement["clips_out"] == len(EXPECTED)
    assert [entry["name"] for entry in manifest["motions"]] == list(EXPECTED)
    assert [row["name"] for row in refinement["records"]] == list(EXPECTED)
    assert [row["name"] for row in support["records"]] == list(EXPECTED)
    assert [row["name"] for row in previous_support["records"]] == list(EXPECTED)
    for report in (refinement, manifest):
        for key in ("teacher_accepted", "hardware_authorized", "deployment_ready"):
            assert report[key] is False
    contacts, force_balance, replays = [], [], []
    for row, force, old_force in zip(refinement["records"], support["records"], previous_support["records"]):
        name, frames = row["name"], EXPECTED[row["name"]]
        assert row == read(REFINED / f"{name}.report.json")
        assert row["frames_in"] == row["frames_out"] == frames and row["frames_dropped"] == 0
        assert row["controlled_joint_count"] == 23 and row["time_scale"] == 1.0
        assert row["root_orientation_changed"] is False
        assert row["output_root_orientation_serialization_error_rad"] <= 2e-7
        assert row["serialized_temporal_audit"]["passed"]
        assert row["causal_terms"]["packets_rebuilt_and_validated"] == frames - 10
        assert row["causal_terms"]["stored_or_sent_to_robot"] is False
        bind(row["output"], row["output_sha256"])
        with np.load(row["output"], allow_pickle=False) as archive:
            assert archive["joint_pos"].shape == (frames, 23)
            assert all(np.isfinite(archive[key]).all() for key in archive.files)
        before, after = row["refinement"]["before"], row["serialized_contact_audit"]
        assert row["contact_and_derivative_constraints_passed"] == after["passed"]
        contacts.append(
            {
                "name": name,
                "frames": frames,
                "before_violated_frames": before["violated_frames"],
                "before_maximum_violation_m": before["maximum_violation_m"],
                "after_violated_frames": after["violated_frames"],
                "after_maximum_violation_m": after["maximum_violation_m"],
                "passed": after["passed"],
                "failure": row["refinement"]["failure"],
                "worst": after["worst"][:2],
                "temporal_audit": row["serialized_temporal_audit"],
                "geometry": {
                    model: {
                        key: value
                        for key, value in report.items()
                        if key in ("frames", "frames_with_floor_overlap", "worst_floor_overlap_m")
                    }
                    for model, report in row["geometry_after"].items()
                },
                "maximum_joint_change_from_original_rad": row["maximum_joint_change_from_original_rad"],
                "maximum_root_offset_per_axis_m": row["maximum_root_offset_per_axis_m"],
            }
        )
        force_row = {"name": name, "models": {}}
        for model in refinement["compiled_models"]:
            current, old = force["models"][model], old_force["models"][model]
            assert current["frames_checked"] == old["frames_checked"] == frames
            assert (
                current["compiled_mjb_sha256"]
                == old["compiled_mjb_sha256"]
                == refinement["compiled_models"][model]
            )
            assert current["torque_limits_nm"] == old["torque_limits_nm"]
            assert current["candidate_floor_gap_tolerance_m"] == old["candidate_floor_gap_tolerance_m"] == 0.002
            force_row["models"][model] = {
                label: {key: value for key, value in item.items() if key.startswith("frames_")}
                for label, item in (("previous", old), ("refined", current))
            }
            if name == "pico_crouch_anchor":
                frame = current["rows"][512]
                points = np.asarray([contact["position_w"][:2] for contact in frame["candidate_contacts"]])
                equations = ConvexHull(points).equations
                com = np.asarray(frame["center_of_mass_w"][:2])
                force_row["models"][model]["static_frame_512"] = {
                    "center_of_mass_xy_m": com.tolist(),
                    "contact_xy_min_m": points.min(axis=0).tolist(),
                    "contact_xy_max_m": points.max(axis=0).tolist(),
                    "maximum_outside_support_hull_halfspace_distance_m": float(
                        (equations[:, :2] @ com + equations[:, 2]).max()
                    ),
                    "maximum_reference_speed": max(abs(value) for value in frame["generalized_velocity"]),
                    "maximum_reference_acceleration": max(
                        abs(value) for value in frame["generalized_acceleration"]
                    ),
                    "force_status": frame["status"],
                    "support_hull_is_diagnostic_not_force_or_contact_complementarity_proof": True,
                }
        force_balance.append(force_row)
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
            {"name": key[0], "measured_start": key[1], "previous": compact_case(old), "refined": compact_case(new)}
        )
    assert all(sha256(Path(path)) == digest for path, digest in IDENTITIES.items())
    result = {
        "kind": "g1_true23_contact_refinement_comparison_v1",
        "clips": len(EXPECTED),
        "frames": sum(EXPECTED.values()),
        "causal_packets_rebuilt_and_validated": sum(EXPECTED.values()) - 10 * len(EXPECTED),
        "contacts": contacts,
        "force_balance": force_balance,
        "replays": replays,
        "diagnostic_pair": new_suite["diagnostic_pair"],
        "independently_rechecked_file_count": len(IDENTITIES),
        "files": IDENTITIES,
        "same_policy_and_actuation_contract_verified": True,
        "original_full_weight_v14_matched_budget_comparison": False,
        "training_or_controller_changed": False,
        "teacher_accepted": False,
        "dynamic_feasibility_proven": False,
        "hardware_authorized": False,
        "deployment_ready": False,
        "limitations": [
            "Geometry restoration retains inferred support hypotheses, not measured contacts.",
            "Force feasibility uses conditional cones and finite-difference reference dynamics, "
            "not contact complementarity.",
            "All eight references and all 23 joints remain; original timing/root orientation are retained.",
            "This fixed-policy reference comparison does not measure retrained tracking or exact 29-DoF parity.",
            "Historical-posture startup/return simulation is not fresh physical state or a Unitree FSM transfer.",
        ],
    }
    with OUTPUT.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"output": str(OUTPUT), "sha256": sha256(OUTPUT), "rechecked_files": len(IDENTITIES)}))


if __name__ == "__main__":
    main()
