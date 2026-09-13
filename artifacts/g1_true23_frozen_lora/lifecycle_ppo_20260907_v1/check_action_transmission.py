"""Same-recorded-state action-transmission diagnostic, not another rollout."""

import hashlib
import json
from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_HARD_LOWER_HARDWARE,
    SAFE_TARGET_HARD_UPPER_HARDWARE,
    safe_target_transform_numpy,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
inputs = {}


def bind(path):
    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    assert inputs.get(str(path), digest) == digest, path
    inputs[str(path)] = digest
    return path


bind(Path(__file__))
for name in ("g1_23dof_contract.py", "g1_23dof_safe_target_transform.py", "g1_true23_sim_acquisition.py"):
    bind(ROOT / "gear_sonic/utils" / name)
rows = []
for update in (1, 50):
    report = json.loads(bind(HERE / f"train100/update_{update}.json").read_text())
    assert report["update"] == update and len(report["episodes"]) == 8
    for row in report["episodes"]:
        result = row["result"]
        with np.load(bind(row["trace_path"]), allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files if key.startswith(("actuation_", "ppo_", "policy_"))}
        n = len(data["actuation_target"])
        assert n == result["completed_active_physics_steps"] and n > 0
        control = np.arange(n) // 10
        sampled_requested = safe_target_transform_numpy(data["policy_raw23"])[1][control].astype(np.float64)
        mean_requested = safe_target_transform_numpy(data["ppo_mean23"].astype(np.float32))[1][control].astype(np.float64)
        np.testing.assert_array_equal(sampled_requested, data["actuation_requested"])
        q, dq, previous = data["actuation_q"], data["actuation_dq"], data["actuation_previous_target"]
        kp, kd = np.asarray(result["gain_kp_hardware"]), np.asarray(result["gain_kd_hardware"])
        cap = .2375 * np.asarray(result["effort_limit_hardware_nm"])
        low = np.maximum(np.maximum(np.asarray(SAFE_TARGET_HARD_LOWER_HARDWARE) + .05,
                                    q + (kd * dq - cap) / kp), previous - .01)
        high = np.minimum(np.minimum(np.asarray(SAFE_TARGET_HARD_UPPER_HARDWARE) - .05,
                                     q + (kd * dq + cap) / kp), previous + .01)
        assert np.all(low <= high)
        sampled_projected = np.clip(sampled_requested, low, high)
        np.testing.assert_array_equal(sampled_projected, data["actuation_target"])
        mean_projected = np.clip(mean_requested, low, high)
        requested_delta = sampled_requested - mean_requested
        projected_delta = sampled_projected - mean_projected
        exact_same = sampled_projected == mean_projected
        clipped = np.abs(sampled_projected - sampled_requested) > 1e-12
        requested_rms = float(np.sqrt(np.mean(requested_delta**2)))
        projected_rms = float(np.sqrt(np.mean(projected_delta**2)))
        record = dict(update=update, case=row["case"]["label"], active_physics_steps=n,
            completed_controls=result["completed_transitions"], requested_controls=result["requested_transitions"],
            sampled_target_clipped_joint_substep_fraction=float(clipped.mean()),
            sampled_and_mean_have_identical_projected_target_fraction=float(exact_same.mean()),
            requested_noise_rms_rad=requested_rms, projected_noise_rms_rad=projected_rms,
            rms_transmission_ratio=projected_rms / requested_rms if requested_rms else None,
            mean_inside_feasible_interval_fraction=float(((mean_requested > low) & (mean_requested < high)).mean()),
            per_joint_identical_projection_fraction=dict(zip(HARDWARE_23_JOINT_NAMES, exact_same.mean(axis=0).tolist(), strict=True)),
            first_substep_identical_projection_fraction=float(exact_same[::10].mean()))
        rows.append(record)
        print(json.dumps({key: value for key, value in record.items() if key != "per_joint_identical_projection_fraction"}))
with (HERE / "action_transmission.json").open("x") as stream:
    json.dump(dict(inputs=inputs, records=rows, evaluated_updates=[1, 50],
        scope="Successful 2 ms substeps in the first and fiftieth sampled training batches; failures remain failures",
        actual_requests_and_projections_reproduced_bit_exactly=True,
        same_state_counterfactual_not_a_closed_loop_noise_ablation=True,
        complete_motion_or_training_cause_proven=False, different_noise_or_action_parameterization_tested=False,
        limits_changed=False, extra_physics_steps=0, extra_policy_calls=0,
        hardware_authorized=False, deployment_ready=False), stream, indent=2)
