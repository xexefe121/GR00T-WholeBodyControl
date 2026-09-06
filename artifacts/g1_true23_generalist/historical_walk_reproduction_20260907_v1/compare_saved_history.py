"""Read-only trace comparison; writes one exclusive evidence receipt."""
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


output = Path(__file__).resolve().parent
root = output.parents[2]
original = root.parent / "GR00T-WholeBodyControl"
old_report_path = original / "artifacts/g1_true23/pico_internet_fullbody_v14_100_eval/model_100/walk001.json"
old_trace_path = old_report_path.with_name("walk001.trajectory.npz")
new_report_path = output / "historical_released_gains.000.original_walk_v14_100.json"
new_trace_path = output / "historical_released_gains.000.original_walk_v14_100.npz"
old = json.loads(old_report_path.read_text())
row = json.loads(new_report_path.read_text())
new = row["result"]
with np.load(old_trace_path, allow_pickle=False) as data:
    old_qpos = data["qpos"].copy()
with np.load(new_trace_path, allow_pickle=False) as data:
    new_qpos = data["qpos"].copy()
    engine_continuous = bool(np.array_equal(data["physics_pre_qpos"][1:], data["physics_post_qpos"][:-1]))
count = min(len(old_qpos), len(new_qpos))
rounded = new_qpos[:count].astype(old_qpos.dtype)
equal_rows = np.all(old_qpos[:count] == rounded, axis=1)
first_difference = None if equal_rows.all() else int(np.flatnonzero(~equal_rows)[0])
receipt = dict(
    kind="historical_original_walk001_saved_trace_reproduction_v1",
    inputs={str(path): sha(path) for path in (old_report_path, old_trace_path, new_report_path, new_trace_path, Path(__file__))},
    same_motion_sha256=old["motion_sha256"] == new["motion_sha256"],
    same_decoder_sha256=old["decoder_sha256"] == row["policy_identity"]["decoder_sha256"],
    historical_gain_profile_reproduced=old["gain_profile"] == "released_retained" and new["actuator_profile"] == "historical_released_gains",
    original_completed=old["completed_transitions"], original_requested=old["requested_transitions"],
    new_completed=new["completed_controls"], new_requested=new["requested_controls"], new_failure=new["failure"],
    original_qpos_shape=list(old_qpos.shape), new_qpos_shape=list(new_qpos.shape),
    initial_qpos_exact_after_original_dtype_rounding=bool(equal_rows[0]),
    all_saved_qpos_exact_after_original_dtype_rounding=len(old_qpos) == len(new_qpos) and bool(equal_rows.all()),
    first_different_saved_boundary=first_difference,
    compared_boundaries=count,
    maximum_absolute_qpos_difference=float(np.max(np.abs(old_qpos[:count].astype(float) - new_qpos[:count]))),
    final_common_boundary_max_abs_difference=float(np.max(np.abs(old_qpos[count-1].astype(float) - new_qpos[count-1]))),
    new_engine_state_contiguous=engine_continuous,
    new_fidelity_screen=new["tracking"]["provisional_reference_landmark_screen_passed"],
    bitwise_reproduction_required_for_bitwise_claim=True,
    historical_source_has_inplace_quaternion_normalization=True,
    new_benchmark_diagnostic_kinematics_does_not_modify_integrated_state=True,
    physical_hardware_equivalence_proven=False,
    hardware_authorized=False, deployment_ready=False,
)
with (output / "saved_history_comparison.json").open("x") as stream:
    json.dump(receipt, stream, indent=2, allow_nan=False)
print(json.dumps(receipt))
