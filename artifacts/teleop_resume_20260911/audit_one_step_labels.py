"""Independent saved feature, action-memory and frozen-teacher branch checks.

No graph calls or native stepping. Does not import the collector's code.
Physical feasibility must first be established by the separate root replay.
"""
import argparse
import json
from pathlib import Path
import sys
import traceback

import numpy as np

from audit_one_step_physics import sha, read, archive, exact, atomic
from branch_label_audit import KEYS, SavedBranchAlgebra, load_original_difference


ROOT = Path(__file__).resolve().parents[2]
NEW = Path("/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911")
OLD = NEW.parent / "sonic23_teleop_six_hour_20260910"
LABEL_KEYS = {
    "state": "endpoint_state", "base_target": "endpoint_base_target",
    "features": "endpoint_features", "teacher_target": "label_fixed_map_target",
    "residual_rad": "label_residual_rad", "teacher_feedback_raw": "teacher_feedback_raw",
    "teacher_tangent": "teacher_tangent",
    "teacher_feedback_correction": "teacher_feedback_correction",
    "teacher_preclip": "teacher_preclip_target",
    "teacher_feedback_clipped": "teacher_feedback_clipped",
    "teacher_native_clipped": "teacher_native_clipped",
    "teacher_plan_control": "successor_plan_control",
    "teacher_plan_local": "successor_plan_local",
    "teacher_replan_boundary": "successor_replan_boundary",
    "teacher_zero_gain": "successor_zero_gain",
}


def main(args):
    assert np.__version__ == "1.26.4"
    import mujoco
    assert mujoco.__version__ == "3.2.3"
    def forbidden(*unused, **unused_named):
        raise RuntimeError("Native stepping prohibited during saved label audit")
    mujoco.mj_step = mujoco.mj_step1 = mujoco.mj_step2 = forbidden
    assert sha(args.collection / "report.json") == args.collection_report_sha256
    assert sha(args.physics_audit / "report.json") == args.physics_report_sha256
    physics = read(args.physics_audit / "report.json")
    assert physics["passed"] and physics["all_saved_samples_and_final291_byteexact"]
    assert physics["nominal_segments"] == physics["policy_segments"] == 3054
    physics_request_path = args.physics_audit / "request.json"
    assert sha(physics_request_path) == physics["request_sha256"]
    physics_request = read(physics_request_path)
    assert physics_request["collection_report_sha256"] == args.collection_report_sha256
    array_dir = args.collection / "data"
    manifest_path = array_dir / "manifest.json"
    assert physics_request["hashes"][str(manifest_path)] == sha(manifest_path)
    segment_path = args.physics_audit / "segments.jsonl"
    assert sha(segment_path) == physics["segments_sha256"]
    segments = {}
    with segment_path.open() as stream:
        for line in stream:
            segment = json.loads(line)
            key = (segment["kind"], segment["row"])
            assert key not in segments
            segments[key] = segment
    assert len(segments) == 6108
    manifest = read(manifest_path)
    paths = [args.collection / "report.json", manifest_path,
             args.physics_audit / "report.json", physics_request_path, segment_path, Path(__file__),
             Path(__file__).with_name("branch_label_audit.py"),
             Path(__file__).with_name("audit_one_step_physics.py"),
             Path(__file__).with_name("branch_audit_oracle.py")]
    data = {}
    for key, spec in manifest["arrays"].items():
        path = (array_dir / spec["path"]).resolve()
        if not path.is_relative_to(array_dir.resolve()):
            raise ValueError("Array outside selected collection")
        assert sha(path) == spec["sha256"], key
        assert physics_request["hashes"][str(path)] == spec["sha256"], key
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        assert list(value.shape) == spec["shape"] and value.dtype == np.dtype(spec["dtype"])
        data[key] = value
        paths.append(path)
    center_path = NEW / "velocity_chord_student_v1/generation/centers.npz"
    assert sha(center_path) == "5b07595d07e262f2ad236565ec62483d600843fe27a9f4a781974dac4b0608c7"
    centers = archive(center_path)
    paths.append(center_path)
    source = NEW / "velocity_chord_student_v1/source_snapshot_v2"
    sys.path.insert(0, str(source))
    from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle, load_motion_override
    from gear_sonic.utils.g1_true23_mpc_student import GoalFeatures
    from gear_sonic.utils.g1_true23_bfm_seed_observations import state_and_terms
    bundle = ROOT / "artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1"
    reference = OLD / "mjbatch_intent_floor_inputs_v1/walk003/reference.npz"
    assert sha(reference) == "2d3b508e1b489499ece5cea8e88a885d9b94aeca82a6a0e8542bc39e690b8033"
    native, contract, original, timeline, manifest_native = load_native_bundle(bundle, "walk003")
    motion, _ = load_motion_override(reference, bundle, "walk003", native,
                                     contract, original, timeline, manifest_native)
    original29 = archive(bundle / "walk003/original29.npz")
    goals = GoalFeatures(motion, original29, contract)
    core = NEW / "bfm_entry250_actual_oracle_v1/source_snapshot_v1/gear_sonic/utils/g1_true23_mjbatch_ilqr_core.py"
    algebra = SavedBranchAlgebra(contract, centers, goals, state_and_terms,
                                  load_original_difference(core))
    paths.extend([core, reference, bundle / "walk003/original29.npz",
                  bundle / "contract.json", bundle / "native_prepared.xml",
                  bundle / "prepared_model_arrays.npz"])
    paths.extend(sorted((source / "gear_sonic/utils").glob("*.py")))
    pins = {str(path): sha(path) for path in paths}
    args.output.mkdir(parents=True, exist_ok=False)
    request = dict(kind="independent_saved_one_control_branch_label_audit",
                   rows=3054, graph_calls=0, native_steps=0, optimizer_updates=0,
                   collection_report_sha256=args.collection_report_sha256,
                   physics_report_sha256=args.physics_report_sha256, hashes=pins)
    atomic(args.output / "request.json", request)
    request_hash = sha(args.output / "request.json")
    controls = np.tile(np.arange(250, 1268, dtype=np.int64), 3)
    datasets = np.repeat(np.arange(3, dtype=np.int64), 1018)
    indices = datasets * 1019 + controls - 250
    for key, expected in (("dataset", datasets), ("start_control", controls),
                          ("successor_control", controls + 1), ("center_index", indices),
                          ("successor_center_index", indices + 1), ("source_frame", controls + 11),
                          ("successor_frame", controls + 12)):
        algebra.exact(data[key], expected, key)
    stats = dict(rows_checked=0, valid_labels=0, failed_branches=0,
                 policy_target_clipped_rows=0, teacher_feedback_clipped_rows=0,
                 teacher_native_clipped_rows=0)
    active = {}
    try:
        for row, index in enumerate(indices):
            index = int(index)
            active = dict(row=np.int64(row), center_index=np.int64(index))
            assert bool(data["nominal_verified"][row])
            algebra.exact(data["nominal_head_output"][row, 0], data["policy_head_delta"][row],
                          "nominal raw head output")
            for phase in ("nominal", "policy"):
                segment = segments[(phase, row)]
                assert segment["dataset"] == int(datasets[row]) and segment["control"] == int(controls[row])
                native_result = segment["oracle"]
                for key in ("valid_steps", "returned_steps", "attempted_steps"):
                    assert int(data[phase + "_" + key][row]) == native_result["physics_steps"]
                expected_status = 1 if native_result["feasible"] else 2
                assert int(data[phase + "_status"][row]) == expected_status
                if phase == "nominal":
                    assert native_result["feasible"] and native_result["physics_steps"] == 10
                else:
                    assert bool(data["label_valid"][row]) == native_result["feasible"]
            for key, center_key in (("incoming_raw_prior", "previous_action"),
                                    ("nominal_features", "features"), ("nominal_state", "state"),
                                    ("nominal_base_action", "base_action"),
                                    ("nominal_base_target", "base_target"),
                                    ("nominal_target", "expert_target"),
                                    ("incoming_history", "history")):
                algebra.exact(data[key][row], centers[center_key][index], key)
            for key in KEYS:
                algebra.exact(data["incoming_history_" + key][row],
                              centers["history_" + key][index], "incoming " + key)
            outgoing = algebra.outgoing(index, data["policy_head_delta"][row])
            for key, expected in zip(("policy_raw_target", "policy_applied_target",
                                      "outgoing_raw_prior", "policy_actual_normalized_action"), outgoing):
                algebra.exact(data[key][row], expected, key)
            target_clip = outgoing[0] != outgoing[1]
            algebra.exact(data["policy_native_target_clipped"][row], target_clip, "policy clip mask")
            stats["policy_target_clipped_rows"] += int(target_clip.any())
            named, flat = algebra.advanced_history(index)
            algebra.exact(data["advanced_history"][row], flat, "advanced flattened history")
            for key in KEYS:
                algebra.exact(data["advanced_history_" + key][row], named[key], "advanced " + key)
            algebra.exact(data["successor_plan_control"][row], centers["plan_control"][index + 1], "plan control")
            algebra.exact(data["successor_plan_local"][row], centers["plan_local"][index + 1], "plan local")
            algebra.exact(data["successor_plan_accepted_update"][row],
                          centers["plan_accepted_update"][index + 1], "plan accepted update")
            algebra.exact(data["successor_replan_boundary"][row],
                          np.bool_(centers["plan_local"][index + 1] == 0), "all-row replan boundary")
            algebra.exact(data["successor_zero_gain"][row],
                          np.bool_(np.all(centers["gain"][index + 1] == 0)), "all-row zero gain")
            if bool(data["label_valid"][row]):
                assert int(data["policy_valid_steps"][row]) == 10
                algebra.exact(data["endpoint_head_output"][row, 0], data["endpoint_head_delta"][row],
                              "endpoint raw head output")
                algebra.exact(data["endpoint_actor_output"][row, 0] * 5,
                              data["endpoint_base_action"][row], "endpoint actor scale")
                q, v = data["policy_qpos"][row, 10], data["policy_qvel"][row, 10]
                active.update(qpos=q, qvel=v, previous_action=outgoing[2])
                expected = algebra.endpoint(index, q, v, outgoing[2], data["endpoint_base_action"][row])
                for key, value in expected.items():
                    algebra.exact(data[LABEL_KEYS[key]][row], value, key)
                proposal = expected["base_target"] + data["endpoint_head_delta"][row]
                applied = np.clip(proposal, algebra.limits[:, 0], algebra.limits[:, 1])
                algebra.exact(data["endpoint_raw_proposal"][row], proposal, "endpoint raw proposal")
                algebra.exact(data["endpoint_applied_target"][row], applied, "endpoint applied target")
                for key in ("features", "base_target", "state", "teacher_target", "residual_rad"):
                    assert np.isfinite(expected[key]).all(), key
                stats["valid_labels"] += 1
                stats["teacher_feedback_clipped_rows"] += int(expected["teacher_feedback_clipped"].any())
                stats["teacher_native_clipped_rows"] += int(expected["teacher_native_clipped"].any())
            else:
                assert int(data["policy_status"][row]) == 2
                for key in ("endpoint_features", "endpoint_state", "endpoint_base_action",
                            "endpoint_base_target", "label_fixed_map_target", "label_residual_rad",
                            "endpoint_latent", "endpoint_head_delta", "endpoint_raw_proposal",
                            "endpoint_applied_target", "teacher_feedback_raw",
                            "teacher_feedback_correction", "teacher_preclip_target",
                            "teacher_tangent", "endpoint_head_output", "endpoint_actor_output"):
                    assert np.isnan(data[key][row]).all(), (row, key)
                stats["failed_branches"] += 1
            stats["rows_checked"] += 1
            if (row + 1) % 250 == 0:
                atomic(args.output / "progress.json", dict(**stats, comparisons=algebra.comparisons))
        assert stats["rows_checked"] == 3054
        assert stats["valid_labels"] == physics["policy_feasible"]
        assert stats["failed_branches"] == physics["policy_failed"]
        # Equal effective float32 features must not silently receive different
        # exact targets or the float32 residuals a later fit would consume.
        seen = {}
        duplicates = target_conflicts = residual_conflicts = 0
        def canonical_bytes(value, dtype):
            value = np.asarray(value, dtype=dtype).copy()
            value[value == 0] = 0
            return value.tobytes()
        groups = [(centers["features"], centers["expert_target"], centers["residual_rad"])]
        valid = np.asarray(data["label_valid"], dtype=bool)
        groups.append((data["endpoint_features"][valid], data["label_fixed_map_target"][valid],
                       data["label_residual_rad"][valid]))
        for features, targets, residuals in groups:
            for feature, target, residual in zip(features, targets, residuals):
                key = canonical_bytes(feature, np.float32)
                label = (canonical_bytes(target, np.float64), canonical_bytes(residual, np.float32))
                if key in seen:
                    duplicates += 1
                    target_conflicts += int(seen[key][0] != label[0])
                    residual_conflicts += int(seen[key][1] != label[1])
                else:
                    seen[key] = label
        compatibility = dict(total_rows=3057 + stats["valid_labels"], unique_feature_inputs=len(seen),
                             duplicate_inputs=duplicates, exact_target_conflicts=target_conflicts,
                             float32_residual_conflicts=residual_conflicts,
                             signed_zero_canonicalized=True, rows_removed=0, targets_averaged=0)
        for path, digest in pins.items():
            assert sha(path) == digest, path
        assert sha(args.output / "request.json") == request_hash
        result = dict(passed=True, **stats, comparisons=algebra.comparisons,
                      graph_calls=0, native_steps=0, optimizer_updates=0,
                      new_labels=0, full_lifecycle_qualification=False,
                      all_saved_features_history_actions_full_teacher_maps_exact=True,
                      compatibility_with_unchanged_3057_nominal_centers=compatibility,
                      request_sha256=request_hash, hashes_unchanged=True)
        atomic(args.output / "report.json", result)
        print(json.dumps(result), flush=True)
    except BaseException as error:
        np.savez_compressed(args.output / "failure_active.npz", **active)
        atomic(args.output / "failure.json", dict(passed=False, **stats,
               comparisons=algebra.comparisons, exception=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("collection", "physics-audit", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--collection-report-sha256", required=True)
    parser.add_argument("--physics-report-sha256", required=True)
    main(parser.parse_args())
