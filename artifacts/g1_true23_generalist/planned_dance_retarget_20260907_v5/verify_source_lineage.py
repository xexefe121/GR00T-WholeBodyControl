"""Cold source identity check: original planned_qpos50, never policy rollout poses."""
from pathlib import Path
import json

import numpy as np
from scipy.spatial.transform import Rotation

from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file

directory = Path(__file__).resolve().parent
report = json.loads((directory / "report.json").read_text())
assert sha256_file(directory / "planned.named29.npz") == report["named_source_sha256"]
assert sha256_file(directory / "adapted.true23.npz") == report["adapted_motion_sha256"]
trace_path = next(Path(path) for path in report["input_bindings"]
                  if path.endswith("happy_dance.cpp_parameters_and_float32_targets.npz"))
assert sha256_file(trace_path) == report["input_bindings"][str(trace_path)]
with np.load(trace_path, allow_pickle=False) as archive:
    planned = archive["planned_qpos50"].copy()
    original_times = archive["command_time_s"].copy()
with np.load(directory / "planned.named29.npz", allow_pickle=False) as archive:
    named = {key: archive[key].copy() for key in archive.files}
with np.load(directory / "adapted.true23.npz", allow_pickle=False) as archive:
    resampled_joints = archive["source_joint_pos_resampled"].copy()
    resampled_root = archive["source_root_pos_w_resampled"].copy()
    resampled_quat = archive["source_root_quat_wxyz_resampled"].copy()
    source_times = archive["source_time_map_s"].copy()
named_qpos = np.column_stack((named["root_pos_w"], named["root_quat_wxyz"], named["joint_pos"]))
np.testing.assert_array_equal(named_qpos, planned)
np.testing.assert_array_equal(named["timestamps_s"], original_times)
np.testing.assert_array_equal(resampled_joints[::2], planned[:, 7:])
np.testing.assert_array_equal(resampled_root[::2], planned[:, :3])
np.testing.assert_allclose(source_times[::2], original_times, atol=1e-12, rtol=0)
source_rotation = Rotation.from_quat(planned[:, [4, 5, 6, 3]])
resampled_rotation = Rotation.from_quat(resampled_quat[::2][:, [1, 2, 3, 0]])
orientation_error = float(np.max((source_rotation * resampled_rotation.inv()).magnitude()))
assert orientation_error < 1e-12
verification_path = directory / "saved_motion_verification.json"
verification = json.loads(verification_path.read_text())
assert verification["passed"] is True
assert sha256_file(directory / "verify_saved_motion.py") == verification["verification_script_sha256"]
receipt = {
    "kind": "g1_true23_v5_raw_planned_source_lineage_verification_v1", "passed": True,
    "raw_source_field": "planned_qpos50", "recorded_policy_poses_used": False,
    "source_frames": 546, "full_named_source_qpos_equal_original": True,
    "all_original_joint_and_root_position_samples_exact_in_source_map": True,
    "all_original_root_orientations_preserved_error_max_rad": orientation_error,
    "raw_trace_sha256": sha256_file(trace_path), "named_source_sha256": report["named_source_sha256"],
    "motion_sha256": report["adapted_motion_sha256"],
    "saved_motion_verification_sha256": sha256_file(verification_path),
    "verification_script_sha256": sha256_file(Path(__file__)),
    "dynamic_feasibility_verified": False, "controller_qualified": False,
    "hardware_authorized": False, "deployment_ready": False,
}
with (directory / "source_lineage_verification.json").open("x") as stream:
    json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
print(json.dumps({"passed": True, "source_frames": 546, "orientation_error_max_rad": orientation_error}))
