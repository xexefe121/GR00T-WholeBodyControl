from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gear_sonic.envs.mjlab.sonic_true23_stage_one_actuation import native_support_pd_step
from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_ACTION_SCALE, ISAACLAB_TO_MUJOCO_DOF
from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE
from gear_sonic.utils.g1_true23_actuation_profile import SIM_CONFIG, NativeSupportActuationProfile
from gear_sonic.utils.g1_true23_projected_controller_state import (
    applied_target_native_numpy,
    applied_target_native_torch,
    synthetic_reset_target_numpy,
    synthetic_reset_target_torch,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def profile():
    return replace(
        NativeSupportActuationProfile.from_sim_config(ROOT / SIM_CONFIG), consistent_controller_state=True
    )


def test_stateful_profile_has_distinct_unqualified_reset_semantics(profile):
    old = replace(profile, consistent_controller_state=False)
    assert old.source_sha256 == profile.source_sha256
    assert old.contract()["kind"] != profile.contract()["kind"]
    assert not profile.contract()["measured_acquisition_reseed_permitted"]
    assert not profile.contract()["reset_reachability_proven"]
    assert not profile.contract()["hardware_authorized"]


def test_applied_feedback_encodes_exact_target_without_second_tanh():
    targets = np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE, dtype=np.float32) + np.linspace(
        -0.3, 0.3, 23, dtype=np.float32
    )
    native = applied_target_native_numpy(targets)
    np.testing.assert_allclose(applied_target_native_torch(torch.tensor(targets)).numpy(), native, rtol=0, atol=0)
    reconstructed = (
        native[list(ISAACLAB_TO_MUJOCO_DOF)] * np.asarray(HARDWARE_23_ACTION_SCALE)
        + SAFE_TARGET_DEFAULT_Q_HARDWARE
    )
    np.testing.assert_allclose(reconstructed, targets, rtol=0, atol=5e-8)


def test_seed_removes_action_independent_initial_infeasibility_without_changing_limits(profile):
    q = torch.tensor(SAFE_TARGET_DEFAULT_Q_HARDWARE, dtype=torch.float64)[None]
    dq = torch.zeros_like(q)
    dq[0, 16] = 10
    assert native_support_pd_step(q, q, q, dq, profile)[2].item()
    target, invalid = synthetic_reset_target_torch(q, dq, profile)
    assert not invalid.item()
    expected = synthetic_reset_target_numpy(q[0], dq[0], profile.kp, profile.kd, profile.effort)
    np.testing.assert_allclose(target[0].numpy(), expected, rtol=0, atol=1e-15)
    applied, torque, bad = native_support_pd_step(q, target, q, dq, profile)
    assert not bad.item()
    assert torch.all((applied - target).abs() <= 0.01 + 1e-12)
    assert torch.all(torque.abs() <= torque.new_tensor(profile.effort) * 0.2375 + 1e-12)


def test_infeasible_and_nonfinite_seed_rows_stay_invalid(profile):
    q = torch.tensor(SAFE_TARGET_DEFAULT_Q_HARDWARE)[None].repeat(3, 1)
    dq = torch.zeros_like(q)
    dq[0, 16] = 1000
    dq[1, 0] = float("nan")
    target, invalid = synthetic_reset_target_torch(q, dq, profile)
    assert invalid.tolist() == [True, True, False]
    assert torch.isfinite(target).all()
    with pytest.raises(ValueError, match="infeasible"):
        synthetic_reset_target_numpy(q[0], dq[0], profile.kp, profile.kd, profile.effort)


def _action(profile):
    from gear_sonic.envs.mjlab.sonic_true23 import _MJLAB_IMPORT_ERROR

    if _MJLAB_IMPORT_ERROR is not None:
        pytest.skip("MJLab unavailable")
    from gear_sonic.envs.mjlab.sonic_true23_stage_one_actuation import StageOneActuationAction

    action = StageOneActuationAction.__new__(StageOneActuationAction)
    action.cfg = SimpleNamespace(profile=profile)
    action._env = SimpleNamespace(device="cpu", num_envs=2)
    q = torch.tensor(SAFE_TARGET_DEFAULT_Q_HARDWARE)[None].repeat(2, 1)
    action._entity = SimpleNamespace(
        data=SimpleNamespace(joint_pos=q, joint_vel=torch.zeros_like(q), encoder_bias=torch.zeros_like(q)),
        set_joint_effort_target=lambda *args, **kwargs: None,
    )
    action._target_ids = torch.arange(23)
    for name in (
        "_raw_actions",
        "_safe_native_actions",
        "_processed_actions",
        "_requested_targets",
        "_previous_targets",
    ):
        setattr(action, name, torch.zeros_like(q))
    action._needs_seed = torch.zeros(2, dtype=torch.bool)
    action.envelope_violation = torch.zeros(2, dtype=torch.bool)
    return action


def test_reset_seeds_after_final_pose_and_preserves_other_environment(profile):
    action = _action(profile)
    action._previous_targets[1] = 0.125
    action._safe_native_actions[1] = 0.5
    action.reset(torch.tensor([0]))
    # Simulates motion reset running after ActionManager.reset.
    action._entity.data.joint_pos[0, 16] = 0.7
    action._entity.data.joint_vel[0, 16] = 10
    expected, invalid = synthetic_reset_target_torch(
        action._entity.data.joint_pos, action._entity.data.joint_vel, profile
    )
    assert not invalid.any()
    history = action.safe_native_action.clone()
    torch.testing.assert_close(action._previous_targets[0], expected[0])
    torch.testing.assert_close(history[0], applied_target_native_torch(expected)[0])
    assert torch.all(action._previous_targets[1] == 0.125)
    assert torch.all(history[1] == 0.5)
    # Subsequent observations must not reseed from moving q/dq.
    action._entity.data.joint_pos[0, 16] += 0.1
    torch.testing.assert_close(action.safe_native_action, history)


def test_process_does_not_report_unexecuted_target_and_bad_seed_latches(profile):
    action = _action(profile)
    action.reset()
    action._entity.data.joint_vel[0, 16] = 1000
    before = action.safe_native_action.clone()
    action.process_actions(torch.ones(2, 23))
    torch.testing.assert_close(action.safe_native_action, before)
    assert action.envelope_violation.tolist() == [True, False]
    action.apply_actions()
    torch.testing.assert_close(action.safe_native_action, applied_target_native_torch(action._previous_targets))
    assert action.envelope_violation[0]


def test_stateful_diagnostic_cannot_bypass_measured_acquisition():
    from gear_sonic.scripts.evaluate_g1_true23_deployment_envelope import run_case

    arguments = dict(
        root=ROOT,
        asset_root=ROOT,
        policy=None,
        motion={},
        kp=None,
        kd=None,
        fraction=1.0,
        joint_scale=None,
        ankle_effort=35.0,
        slew_rate=5.0,
        initial_state="measured",
        stateful_native_controller=True,
    )
    with pytest.raises(ValueError, match="requires active effort"):
        run_case(**arguments)
    with pytest.raises(ValueError, match="preceding balance controller"):
        run_case(**arguments, project_active_effort=True)


def test_real_simulator_history_tracks_executed_target(profile, monkeypatch):
    from gear_sonic.scripts import evaluate_g1_true23_deployment_envelope as envelope
    from gear_sonic.utils.g1_true23_clean_mujoco_teleop import NATIVE_TO_IL29

    assets = ROOT.parent / "GR00T-WholeBodyControl"
    motion_path = (
        assets / "artifacts/g1_true23/sonic_library_true23_happy_physical_reference_v1/happy_dance.true23.npz"
    )
    if not motion_path.is_file():
        pytest.skip("local simulation motion unavailable")
    with np.load(motion_path, allow_pickle=False) as archive:
        motion = {key: archive[key].copy() for key in archive.files}
    histories, targets = [], []
    original_projection = envelope.effort_feasible_target

    def project(*args, **kwargs):
        result = original_projection(*args, **kwargs)
        targets.append(result.copy())
        return result

    def infer(encoder, history):
        histories.append(history.copy())
        return np.zeros(23, dtype=np.float32), None

    monkeypatch.setattr(envelope, "effort_feasible_target", project)
    report, _ = envelope.run_case(
        root=ROOT,
        asset_root=assets,
        policy=SimpleNamespace(infer=infer),
        motion=motion,
        kp=np.asarray(profile.kp),
        kd=np.asarray(profile.kd),
        fraction=1.0,
        joint_scale=np.ones(23),
        ankle_effort=35.0,
        slew_rate=5.0,
        initial_state="reference",
        maximum_steps=2,
        project_active_effort=True,
        stateful_native_controller=True,
    )
    assert report["completed_transitions"] == 2 and len(targets) == 20
    seed = synthetic_reset_target_numpy(
        motion["joint_pos"][10], motion["joint_vel"][10], profile.kp, profile.kd, profile.effort
    )
    initial_actions = histories[0][610:900].reshape(10, 29)[:, NATIVE_TO_IL29]
    np.testing.assert_array_equal(initial_actions, np.tile(applied_target_native_numpy(seed), (10, 1)))
    second_actions = histories[1][610:900].reshape(10, 29)[:, NATIVE_TO_IL29]
    np.testing.assert_array_equal(second_actions[-1], applied_target_native_numpy(targets[9]))
    assert np.count_nonzero(second_actions[-1]) > 0  # Requested raw actions were all zero.
    assert report["synthetic_reference_controller_reset"]
    assert not report["measured_controller_target_reseeded_after_acquisition"]
