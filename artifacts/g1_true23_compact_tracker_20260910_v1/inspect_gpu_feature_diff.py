"""Inspect saved initial GPU/CPU compact observations without simulation."""

from pathlib import Path
import json
import numpy as np

base = Path(__file__).resolve().parent
for name in ("walk002", "walk003", "pico"):
    gpu = dict(np.load(base / "gpu_replay1000_v1" / f"{name}.npz", allow_pickle=False))
    cpu = dict(np.load(base / "eval1000_recovered_v1" / name / "attempts.npz", allow_pickle=False))
    left, right = gpu["features"][0], cpu["native_controller165"][0]
    groups = [("angular_velocity", 0, 3), ("joint_position", 3, 26), ("joint_velocity", 26, 49),
              ("previous_action", 49, 72), ("gravity", 72, 75), ("goal_q", 75, 98), ("goal_dq", 98, 121),
              ("root_feedback", 121, 130), ("root_orientation", 130, 136), ("vr", 136, 157), ("feet", 157, 163), ("height", 163, 165)]
    print(json.dumps({"name": name, "initial_feature_differences": {key: {"max_abs": float(np.max(np.abs(left[start:end] - right[start:end]))),
        "gpu": left[start:end].tolist(), "cpu": right[start:end].tolist()} for key, start, end in groups if np.max(np.abs(left[start:end] - right[start:end])) > 1e-6}}))
