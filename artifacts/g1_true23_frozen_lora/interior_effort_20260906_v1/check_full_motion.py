"""Extend independent evidence checks to every standing-adapter motion run."""

import json
from pathlib import Path

import numpy as np

from gear_sonic.scripts.record_g1_sonic_original29_baseline import dump
from gear_sonic.utils.g1_sonic_cpp_observation_trace import audit_engine_trace
from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
original = json.loads((HERE / "integrity_report.json").read_text())
report = json.loads((HERE / "standing_lora_full_motion/report.json").read_text())
inputs = dict(original["inputs"])


def bind(path, expected=None):
    path = Path(path).resolve(strict=True)
    digest = file_sha256(path)
    if digest != inputs.get(str(path), digest) or (expected is not None and digest != expected):
        raise ValueError(f"changed full-motion evidence: {path}")
    inputs[str(path)] = digest
    return path


for path, digest in report["inputs"].items():
    bind(path, digest)
bind(HERE / "integrity_report.json")
bind(HERE / "standing_lora_full_motion/report.json")
bind(Path(__file__))
config = json.loads((ROOT / "gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json").read_text())
effort = np.asarray(config["physics"]["effort_limit_hardware_nm"])
assert effort[[4, 5, 10, 11]].tolist() == [35] * 4
rows = []
assert len(report["records"]) == 9
assert sum(bool(row.get("not_executed")) for row in report["records"]) == 1
for record in report["records"]:
    if record.get("not_executed"):
        assert record["name"] == "elbow_crawling"
        continue
    with np.load(bind(record["trace_path"]), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    with np.load(bind(record["source_motion_path"], record["source_motion_sha256"]), allow_pickle=False) as source:
        assert record["result"]["requested_transitions"] == len(source["joint_pos"]) - 11
    engine = audit_engine_trace(arrays)
    assert engine["passed"]
    for kind in ("qpos", "qvel"):
        np.testing.assert_array_equal(arrays[f"physics_pre_{kind}"][1:], arrays[f"physics_post_{kind}"][:-1])
        assert (
            np.isfinite(arrays[f"physics_pre_{kind}"]).all() and np.isfinite(arrays[f"physics_post_{kind}"]).all()
        )
    np.testing.assert_array_equal(arrays["physics_effort"], arrays["physics_generalized_actuator_force"])
    peak = float(np.max(np.abs(arrays["physics_effort"]) / (0.95 * 0.25 * effort)))
    assert peak <= 1 + 1e-10
    assert not record["result"]["motion_fidelity"]["passed"]
    assert (
        record["result"]["compiled_native_model_sha256"]
        == "1f616be811e72988f4764f74b8c2eb7f57b5ee87873ee36204bd882ff9cf96d7"
    )
    assert int(np.sum(arrays["physics_phase"] == 1)) == record["result"]["completed_active_physics_steps"]
    rows.append(
        dict(
            name=record["name"],
            historical_start=record["historical_start"],
            physics_steps=engine["physics_steps_checked"],
            actual_time=engine["actual_final_engine_time_s"],
            maximum_outer_effort_ratio=peak,
        )
    )
for path in list(inputs):
    bind(path)
dump(
    HERE / "full_integrity_report.json",
    {
        "kind": "g1_true23_headroom_standing_full_motion_integrity_v1",
        "inputs": inputs,
        "unique_bound_file_count": len(inputs),
        "mismatches": [],
        "previous_trace_count": len(original["traces"]),
        "new_traces": rows,
        "total_continuous_trace_count": len(original["traces"]) + len(rows),
        "full_source_requests_preserved": True,
        "all_actual_clocks_continuous": True,
        "all_warning_counts_zero": True,
        "full_motion_qualified": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    },
)
print(
    json.dumps(
        {
            "bound_files": len(inputs),
            "total_continuous_traces": len(original["traces"]) + len(rows),
            "mismatches": 0,
        }
    ),
    flush=True,
)
