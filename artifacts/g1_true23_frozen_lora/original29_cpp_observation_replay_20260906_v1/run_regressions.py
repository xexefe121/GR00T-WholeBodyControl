"""Offline regression suite; retain the two legacy asset-root substitutions."""

import json
from pathlib import Path

import pytest

from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256

ROOT = Path.cwd()
ASSETS = Path("/mnt/z/codex/GR00T-WholeBodyControl")
OUTPUT = Path(__file__).resolve().parent
MODULES = """
test_g1_true23_deployment_envelope
test_g1_true23_sim_acquisition
test_g1_true23_projected_controller_state
test_g1_true23_predictive_target_filter
test_retime_g1_true23_sonic_reference
test_g1_true23_stage_one_actuation
test_g1_true23_actuation_reset_feasibility
test_train_g1_23dof_mjlab_frozen_lora
test_g1_true23_exploration_rollout
test_g1_true23_reference_floor
test_g1_true23_reference_support
test_g1_true23_pico_collision_grounding
test_g1_true23_stance_retarget
test_g1_true23_training_replay_parity
test_g1_true23_clean_mujoco_teleop
test_g1_true23_pico_fullbody_motion
test_g1_true23_requested_projection_cost
test_g1_true23_paired_envelope_suite
test_g1_true23_contact_trajectory
test_g1_true23_force_trajectory
test_g1_true23_contact_patch
test_g1_true23_box_qp
test_g1_true23_contact_force_optimizer
test_g1_true23_reference_lineage
test_g1_true23_original_sonic_parity
test_g1_true23_original_task_trajectory
test_g1_sonic_original29_trace
test_g1_sonic_cpp_parameters
test_retarget_g1_true23_original29_trace
test_g1_sonic_hand_collision_variant
test_g1_true23_hand_frame_tasks
test_g1_sonic_cpp_observations
""".split()
paths = [ROOT / "gear_sonic/tests" / f"{name}.py" for name in MODULES]
inputs = {str(path): file_sha256(path) for path in [*paths, Path(__file__), OUTPUT / "report.json"]}


class OriginalAssetRoot:
    def pytest_collection_modifyitems(self, items):
        seen = set()
        for item in items:
            module = item.module
            if module in seen or not module.__file__.endswith(
                ("test_g1_true23_clean_mujoco_teleop.py", "test_g1_true23_pico_fullbody_motion.py")
            ):
                continue
            seen.add(module)
            for name, value in list(vars(module).items()):
                if isinstance(value, Path) and value.is_relative_to(ROOT):
                    setattr(module, name, ASSETS / value.relative_to(ROOT))


code = int(
    pytest.main([*map(str, paths), "-q", f"--junitxml={OUTPUT / 'regression.xml'}"], plugins=[OriginalAssetRoot()])
)
for path, digest in inputs.items():
    if file_sha256(path) != digest:
        raise ValueError(f"regression input changed: {path}")
inputs[str(OUTPUT / "regression.xml")] = file_sha256(OUTPUT / "regression.xml")
with (OUTPUT / "regression_report.json").open("x") as stream:
    json.dump(
        {
            "exit_code": code,
            "modules": MODULES,
            "inputs": inputs,
            "original_asset_root": str(ASSETS),
            "substituted_module_asset_roots": [
                "test_g1_true23_clean_mujoco_teleop",
                "test_g1_true23_pico_fullbody_motion",
            ],
            "hardware_authorized": False,
            "deployment_ready": False,
        },
        stream,
        indent=2,
    )
raise SystemExit(code)
