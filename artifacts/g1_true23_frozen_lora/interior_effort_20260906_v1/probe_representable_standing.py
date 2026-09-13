"""Test explicit safe-boundary standing teacher; no dance substitution."""

import json
from pathlib import Path
import sys

import numpy as np

from gear_sonic.scripts.record_g1_sonic_original29_baseline import dump
from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256
from gear_sonic.utils.g1_true23_standing_bootstrap import collect_standing_teacher

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ASSETS = ROOT.parent / "GR00T-WholeBodyControl"
OUTPUT = HERE / "representable_standing"
source = json.loads((HERE / "stationary/report.json").read_text())
inputs = dict(source["inputs"])


def bind(path):
    path = Path(path).resolve(strict=True)
    digest = file_sha256(path)
    if inputs.get(str(path), digest) != digest:
        raise ValueError(f"standing source changed: {path}")
    inputs[str(path)] = digest
    return path


bind(HERE / "stationary/report.json")
bind(Path(__file__))
for name, module in list(sys.modules.items()):
    path = getattr(module, "__file__", None)
    if name.startswith("gear_sonic.") and path and Path(path).suffix == ".py":
        bind(path)
policy_path = bind(
    ASSETS / "artifacts/external/unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx"
)
with np.load(bind(HERE / "stationary/stationary_reference.npz"), allow_pickle=False) as archive:
    motion = {key: archive[key].copy() for key in archive.files}
OUTPUT.mkdir(exist_ok=False)
dump(OUTPUT / "started.json", {"inputs": inputs, "hardware_authorized": False, "deployment_ready": False})
perturb = np.zeros(23)
perturb[[0, 6]] = 0.01
perturb[[3, 9]] = -0.01
perturb[[4, 10]] = 0.01
records = []
for name, projected, delta, role in (
    ("original_nominal", False, None, "control_not_training"),
    ("bounded_nominal", True, None, "standing_train"),
    ("bounded_plus", True, perturb, "standing_train"),
    ("bounded_minus", True, -perturb, "standing_train"),
    ("bounded_holdout", True, perturb * 0.5, "held_out_episode"),
):
    result, arrays = collect_standing_teacher(
        root=ROOT,
        asset_root=ASSETS,
        policy_path=policy_path,
        motion=motion,
        project_request=projected,
        initial_joint_delta=delta,
    )
    if result["compiled_native_model_sha256"] != source["records"][0]["result"]["compiled_native_model_sha256"]:
        raise ValueError("standing teacher and SONIC compiled models differ")
    if name == "original_nominal":
        with np.load(HERE / "stationary/standing_compatibility_actor.npz", allow_pickle=False) as old:
            matches = {}
            for key in (
                "qpos",
                "encoder267",
                "history930",
                "physics_pre_qpos",
                "physics_post_qpos",
                "physics_pre_qvel",
                "physics_post_qvel",
            ):
                np.testing.assert_array_equal(arrays[key], old[key])
                matches[key] = True
            result["previous_teacher_trace_bit_exact_after_model_limit_alignment"] = matches
    path = OUTPUT / f"{name}.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    bind(path)
    records.append({"name": name, "role": role, "arrays": str(path), "result": result})
    dump(OUTPUT / f"{name}.json", records[-1])
    bind(OUTPUT / f"{name}.json")
    print(
        json.dumps(
            {
                "name": name,
                "completed": result["completed_transitions"],
                "usable_for_standing_only": result["standing_only_labels_usable"],
                "changed_joint_requests": result["unrepresentable_requested_joint_controls"],
                "maximum_request_change_rad": result["maximum_request_change_rad"],
                "failure": result["failure"],
            }
        ),
        flush=True,
    )
for path in list(inputs):
    bind(path)
dump(
    OUTPUT / "report.json",
    {
        "kind": "g1_true23_representable_standing_teacher_comparison_v1",
        "inputs": inputs,
        "records": records,
        "held_out_episode_separate_from_training": True,
        "full_motion_teacher_accepted": False,
        "full_motion_suite_not_replaced": True,
        "hardware_authorized": False,
        "deployment_ready": False,
    },
)
