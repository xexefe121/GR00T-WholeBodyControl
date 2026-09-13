"""Local synthetic reference-only timing evidence, no transport or actuators."""
from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.utils import g1_23dof_task_space_retarget as ik
from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE
from gear_sonic.utils.g1_true23_generalist_causal import CausalNative23Retargeter
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256


def main():
    root = Path(__file__).resolve().parents[3]
    output = Path(__file__).with_name("runtime_report.json")
    if output.exists():
        raise FileExistsError(output)
    source, target = ik.load_models(root / ik.DEFAULT_SOURCE_MODEL, root / ik.DEFAULT_TARGET_MODEL)
    paths = [Path(__file__), Path(ik.__file__), root / "gear_sonic/utils/g1_true23_generalist_causal.py"]
    bindings = {str(path): sha256_file(path) for path in paths}
    names = ik._model_layout(source).joint_names
    defaults = dict(zip(HARDWARE_23_JOINT_NAMES, SAFE_TARGET_DEFAULT_Q_HARDWARE, strict=True))
    rows = []
    for scenario in ("static_neutral", "small_source_shoulder_excursion"):
        core = CausalNative23Retargeter(source, target)
        core.initialize(joint_names=HARDWARE_23_JOINT_NAMES, joint_pos=np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE), joint_vel=np.zeros(23), timestamp_s=0.0)
        samples = []
        for tick in range(1, 101):
            pose = np.asarray([defaults.get(name, 0.0) for name in names])
            if scenario != "static_neutral":
                pose[names.index("left_shoulder_roll_joint")] += 0.03 * math.sin(tick * 0.02 * 2 * math.pi)
            result = core.step(joint_names=names, joint_pos=pose, root_pos_w=np.asarray([0.0, 0.0, 0.8]), root_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]), contact_flags=np.asarray([True, True]), timestamp_s=tick * 0.02, now_s=tick * 0.02)
            samples.append({"tick": tick, "accepted": result.accepted, "status": result.status, "diagnostics": result.diagnostics})
            if not result.accepted:
                break
        rows.append({"scenario": scenario, "requested_frames": 100, "completed_frames": sum(row["accepted"] for row in samples), "timing": core.timing_summary(), "samples": samples})
    for path, expected in bindings.items():
        if sha256_file(Path(path)) != expected:
            raise ValueError(f"source changed during measurement: {path}")
    report = {"kind": "g1_true23_generalist_causal_synthetic_reference_runtime", "source_bindings": bindings, "compiled_model_sha256": {"source": compiled_model_sha256(source), "target": compiled_model_sha256(target)}, "numpy_version": np.__version__, "mujoco_version": mujoco.__version__, "records": rows, "simulation_physics_stepped": False, "trained_policy_executed": False, "scheduled_50hz_execution_verified": False, "live_teleop_ready": False, "hardware_authorized": False}
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps([{key: row[key] for key in ("scenario", "requested_frames", "completed_frames", "timing")} for row in rows]))
    print(output)


if __name__ == "__main__":
    main()
