from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from gear_sonic.scripts.g1_true23_reference_bank_campaign import validate_bank_layout
from gear_sonic.tests.test_g1_true23_registered_bank_reference import registered_materials
from gear_sonic.utils import g1_23dof_task_space_retarget as ik
from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE
from gear_sonic.utils.g1_true23_contact_bank_reference import CONTACT_PROFILE, audit_contact_motion
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256


def contact_materials():
    _, previous, bank = registered_materials()
    current = deepcopy(previous)
    current["derived_arrays_sha256"] = "contact-bank"
    current["source_reference_conditioning"] = CONTACT_PROFILE
    for old, new in zip(previous["derived_spans"]["spans"], current["derived_spans"]["spans"], strict=True):
        new["derived_arrays_sha256"] += "-contact"
        new["registered_source_arrays_sha256"] += "-contact"
        new["source_reference_conditioning"] = dict(
            profile=CONTACT_PROFILE,
            original_source_arrays_sha256=old["source_arrays_sha256"],
            original_registered_source_arrays_sha256=old["registered_source_arrays_sha256"],
            final_registered_source_arrays_sha256=new["registered_source_arrays_sha256"],
            final_source_start_registration=new["source_start_registration"],
            geometry=dict(independent_serialized_geometry_passed=True),
            all_original_frames_retained_without_retiming=True,
            raw_original_fidelity_acceptance_inherited=False,
            same_reference_benchmark_claimed=False,
            measured_robot_state_used=False,
            dynamic_feasibility_proven=False,
            hardware_authorized=False,
            deployment_ready=False,
            remaining_conditional_force_support_failures=1,
            remaining_conditional_effort_failures=0,
        )
    return previous, current, bank


def test_changed_contact_reference_requires_explicit_branch_and_retains_failures():
    old, new, bank = contact_materials()
    with pytest.raises(ValueError, match="explicit changed-reference"):
        validate_bank_layout(old, new, bank, allow_expansion=False)
    result = validate_bank_layout(old, new, bank, allow_expansion=False, allow_contact_conditioning=True)
    assert result["reference_arrays_changed"]
    assert not result["same_reference_resume_claimed"]
    assert not result["raw_original_fidelity_acceptance_inherited"]
    assert set(result["remaining_conditional_force_support_failures"].values()) == {1}
    assert not result["deployment_ready"]
    assert validate_bank_layout(deepcopy(new), new, bank, allow_expansion=False)["same_reference_resume_claimed"]
    with pytest.raises(ValueError, match="explicit changed-reference"):
        validate_bank_layout(new, new, bank, allow_expansion=False, allow_contact_conditioning=True)


@pytest.mark.parametrize(
    "mutation",
    ["raw", "crop", "ownership", "phase", "support", "fidelity", "state", "geometry", "repair", "attitude"],
)
def test_contact_transition_cannot_hide_reference_or_qualification_change(mutation):
    old, new, bank = contact_materials()
    row = new["derived_spans"]["spans"][0]
    proof = row["source_reference_conditioning"]
    if mutation == "raw":
        row["source_arrays_sha256"] = "different"
    elif mutation == "crop":
        row["original_source_indices_requested"].pop()
    elif mutation == "ownership":
        row["ownership"] = {"split": "test"}
    elif mutation == "phase":
        row["timeline"]["total_requested_controls"] += 1
    elif mutation == "support":
        proof.pop("remaining_conditional_force_support_failures")
    elif mutation == "fidelity":
        proof["raw_original_fidelity_acceptance_inherited"] = True
    elif mutation == "state":
        proof["measured_robot_state_used"] = True
    elif mutation == "geometry":
        proof["geometry"]["independent_serialized_geometry_passed"] = False
    elif mutation == "repair":
        row["timeline"].pop("generated_reference_repair")
    else:
        proof["original_registered_source_arrays_sha256"] = "changed"
    with pytest.raises(ValueError):
        validate_bank_layout(old, new, bank, allow_expansion=False, allow_contact_conditioning=True)


@pytest.fixture
def grounded_motion():
    model = mujoco.MjModel.from_xml_path(
        str(Path(__file__).resolve().parents[1] / "data/robots/g1/g1_23dof_rev_1_0.xml")
    )
    poses = np.tile(model.qpos0, (12, 1))
    poses[:, 2] = 0.78
    poses[:, 7:] = SAFE_TARGET_DEFAULT_Q_HARDWARE
    from gear_sonic.scripts.diagnose_g1_true23_stance_foot_cleanup import sole_gaps

    poses[:, 2] -= float(sole_gaps(model, poses).min()) - 0.0005

    def build(p):
        return ik.build_mjlab_motion_arrays(
            model,
            SimpleNamespace(root_pos_w=p[:, :3], root_quat_wxyz=p[:, 3:7], joint_pos_hardware=p[:, 7:], fps=50),
        )

    conditioned = build(poses)
    poses[:, 2] -= 0.004
    return model, build(poses), conditioned


def test_actual_geometry_checked_without_optimizer_or_model_changes(grounded_motion):
    model, original, conditioned = grounded_motion
    identity = compiled_model_sha256(model)
    result = audit_contact_motion(model, original, conditioned)
    assert result["independent_serialized_geometry_passed"]
    assert result["samples_100hz"] == 23
    assert result["maximum_ankle_translation_m"] == pytest.approx(0.004, abs=1e-7)
    assert not result["dynamics_or_contact_complementarity_proven"]
    assert compiled_model_sha256(model) == identity


@pytest.mark.parametrize(
    "mutation", ["root", "waist", "quaternion", "fps", "velocity", "acceleration", "nan", "floor", "upper"]
)
def test_serialized_reference_mutations_rejected_independently(grounded_motion, mutation):
    model, original, conditioned = grounded_motion
    bad = deepcopy(conditioned)
    if mutation == "root":
        bad["body_pos_w"][:, :, 0] += 0.04
    elif mutation == "waist":
        bad["joint_pos"][:, 12] += 0.001
    elif mutation == "quaternion":
        bad["body_quat_w"][:, 0, 1] = 0.001
    elif mutation == "fps":
        bad["fps"][:] = 25
    elif mutation == "velocity":
        bad["joint_vel"][2, 0] += 0.1
    elif mutation == "acceleration":
        bad["body_pos_w"][2, :, 0] += 0.001
    elif mutation == "nan":
        bad["body_lin_vel_w"][0, 0, 0] = np.nan
    elif mutation == "floor":
        bad["body_pos_w"][:, :, 2] -= 0.003
    else:
        bad["joint_pos"][:, 13] += 0.4
    with pytest.raises(ValueError):
        audit_contact_motion(model, original, bad)
