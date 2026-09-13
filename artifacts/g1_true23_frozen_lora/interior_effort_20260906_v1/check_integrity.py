"""Independent saved-array audit of headroom and standing-only experiments."""

import json
from pathlib import Path

import numpy as np

from gear_sonic.scripts.fit_g1_true23_standing_lora import validate_standing_labels
from gear_sonic.scripts.record_g1_sonic_original29_baseline import dump
from gear_sonic.utils.g1_sonic_cpp_observation_trace import audit_engine_trace
from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256
from gear_sonic.utils.g1_true23_actuation_profile import NativeSupportActuationProfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
inputs, reports, traces = {}, [], []


def bind(path, expected=None):
    path = Path(path).resolve(strict=True)
    digest = file_sha256(path)
    if digest != inputs.get(str(path), digest) or (expected is not None and digest != expected):
        raise ValueError(f"changed bound evidence: {path}")
    inputs[str(path)] = digest
    return path


bind(Path(__file__))
profile = NativeSupportActuationProfile.from_sim_config(
    bind(ROOT / "gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json")
)
for relative in (
    "report.json",
    "stationary/report.json",
    "representable_standing/report.json",
    "standing_lora500/report.json",
):
    path = bind(HERE / relative)
    report = json.loads(path.read_text())
    for key, digest in report["inputs"].items():
        bind(key, digest)
    if report.get("hardware_authorized") is not False or report.get("deployment_ready") is not False:
        raise ValueError("diagnostic evidence incorrectly promoted")
    reports.append((path, report))
    for record in report["records"]:
        if record.get("not_executed"):
            assert record["name"] == "elbow_crawling"
            continue
        name = record.get("label", record["name"])
        archive = record.get("trace_path", record.get("arrays", str(path.parent / f"{record['name']}.npz")))
        with np.load(bind(archive), allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        engine = audit_engine_trace(arrays)
        assert engine["passed"]
        for kind in ("qpos", "qvel"):
            before, after = [arrays[f"physics_{phase}_{kind}"] for phase in ("pre", "post")]
            assert np.isfinite(before).all() and np.isfinite(after).all()
            np.testing.assert_array_equal(before[1:], after[:-1])
        effort_key = "physics_effort" if "physics_effort" in arrays else "physics_requested_effort"
        actual_key = (
            "physics_generalized_actuator_force"
            if "physics_generalized_actuator_force" in arrays
            else "physics_actual_effort"
        )
        np.testing.assert_array_equal(arrays[effort_key], arrays[actual_key])
        peak_ratio = float(np.max(np.abs(arrays[effort_key]) / (0.95 * 0.25 * np.asarray(profile.effort))))
        assert peak_ratio <= 1 + 1e-10
        if "physics_phase" in arrays:
            result = record["result"]
            assert int(np.sum(arrays["physics_phase"] == 1)) == result["completed_active_physics_steps"]
            assert len(arrays["interior_outer_effort_ratio"]) == result["completed_active_physics_steps"]
        if record.get("role") in ("standing_train", "held_out_episode"):
            validate_standing_labels(arrays)
        traces.append(
            dict(
                name=name,
                report=str(path),
                physics_steps=engine["physics_steps_checked"],
                actual_time=engine["actual_final_engine_time_s"],
                maximum_outer_effort_ratio=peak_ratio,
            )
        )

headroom = reports[0][1]
assert len(headroom["records"]) == 36
assert sum(bool(row.get("not_executed")) for row in headroom["records"]) == 4
assert headroom["full_eight_clip_qualification"] is False and headroom["default_candidate_selected"] is False
for row in headroom["records"]:
    if row.get("not_executed"):
        continue
    assert row["result"]["completed_transitions"] < row["result"]["requested_transitions"]
    assert not row["result"]["motion_fidelity"]["passed"]
    if row.get("baseline_array_equivalence"):
        old_dir = ROOT / "artifacts/g1_true23_frozen_lora/original29_hand_frame_envelope_20260906_v1"
        state = "historical_measured" if row["historical_start"] else "reference"
        old_file = (
            old_dir
            / f"original_masks_{row['name']}_{state}"
            / (
                "configured_sim_fraction1_ankle35_slew5_"
                + ("measured" if row["historical_start"] else "reference")
                + ".npz"
            )
        )
        with (
            np.load(bind(old_file), allow_pickle=False) as old,
            np.load(bind(row["trace_path"]), allow_pickle=False) as new,
        ):
            for key in old.files:
                np.testing.assert_array_equal(new[key], old[key])
for path in list(inputs):
    bind(path)
dump(
    HERE / "integrity_report.json",
    {
        "kind": "g1_true23_headroom_standing_saved_evidence_integrity_v1",
        "inputs": inputs,
        "unique_bound_file_count": len(inputs),
        "mismatches": [],
        "traces": traces,
        "all_actual_engine_clocks_continuous": True,
        "all_warning_counters_zero": True,
        "all_recorded_effort_inside_unchanged_outer_envelope": True,
        "full_motion_qualified": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    },
)
print(json.dumps({"bound_files": len(inputs), "continuous_traces": len(traces), "mismatches": 0}), flush=True)
