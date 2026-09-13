"""Independently rehash bound evidence and inspect retained trace boundaries."""

import json
from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_sonic_cpp_observation_trace import audit_engine_trace
from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256

OUTPUT = Path(__file__).resolve().parent
source_path = OUTPUT / "report.json"
source = json.loads(source_path.read_text())
inputs = {
    **source["inputs"],
    str(source_path): file_sha256(source_path),
    str(Path(__file__)): file_sha256(__file__),
}
assert source["all_requested_cases_recorded"] is True
assert source["all_original_clips_completed"] is False
assert all(source[flag] is False for flag in ("teacher_accepted", "hardware_authorized", "deployment_ready"))
assert [(r["model_variant"], r["name"]) for r in source["records"]] == [
    (model, name)
    for model in ("original_masks", "hand_collisions")
    for name in ("hand_crawling", "elbow_crawling", "happy_dance")
]
records, dances = [], {}
for row in source["records"]:
    with np.load(row["trace_path"], allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    assert all(np.isfinite(value).all() for value in arrays.values())
    engine = audit_engine_trace(arrays)
    assert engine == row["details"]["actual_engine_audit"] and engine["passed"]
    np.testing.assert_array_equal(arrays["physics_pre_qpos"][1:], arrays["physics_post_qpos"][:-1])
    np.testing.assert_array_equal(arrays["physics_pre_qvel"][1:], arrays["physics_post_qvel"][:-1])
    np.testing.assert_array_equal(arrays["encoder_outputs"], arrays["decoder_inputs"][:, :64])
    np.testing.assert_array_equal(arrays["raw_actions"], arrays["decoder_outputs"][: len(arrays["raw_actions"])])
    np.testing.assert_array_equal(arrays["decoder_inputs"][0, 964:991].reshape(9, 3), np.tile([0, 0, 1], (9, 1)))
    final = arrays["decoder_outputs"][-1]
    index = int(np.argmax(np.abs(final)))
    assert len(arrays["qpos"]) == row["details"]["frames_completed"]
    assert len(arrays["physics_post_qpos"]) == 10 * len(arrays["qpos"])
    with np.load(row["same_state_trace_path"], allow_pickle=False) as same:
        assert len(same["decoder_inputs"]) == row["same_state_comparison"]["inference_calls_compared"]
    records.append(
        {
            "model": row["model_variant"],
            "clip": row["name"],
            "arrays_retained": len(arrays),
            "physics_steps": len(arrays["physics_post_qpos"]),
            "control_steps": len(arrays["qpos"]),
            "inference_calls": len(arrays["decoder_inputs"]),
            "last_action_largest_isaac_index": index,
            "last_action_at_index": float(final[index]),
            "engine_audit_passed": True,
        }
    )
    if row["name"] == "happy_dance":
        dances[row["model_variant"]] = arrays
first, second = dances["original_masks"], dances["hand_collisions"]
different = np.flatnonzero(np.any(first["physics_post_qpos"] != second["physics_post_qpos"], axis=1))
for path, digest in inputs.items():
    assert file_sha256(path) == digest, f"bound evidence changed: {path}"
report = {
    "kind": "g1_sonic_cpp_recorded_source_integrity_audit_v1",
    "inputs": inputs,
    "records": records,
    "all_bound_files_match": True,
    "unique_bound_files": len(inputs),
    "cross_model_dance_first_differing_physics_step": int(different[0]) if len(different) else None,
    "physical_damping_cause_diagnosed": False,
    "teacher_accepted": False,
    "hardware_authorized": False,
    "deployment_ready": False,
}
with (OUTPUT / "integrity_report.json").open("x") as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
print(json.dumps({key: value for key, value in report.items() if key != "inputs"}), flush=True)
