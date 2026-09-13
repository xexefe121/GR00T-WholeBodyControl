import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from gear_sonic.envs.mjlab.sonic_true23_nominal_scene import (
    configure_nominal_scene,
    nominal_scene_contract,
    verify_nominal_scene,
)
from gear_sonic.utils.g1_true23_native_model_actuation import NativeModelActuationProfile
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT.parent / "GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml"
PHYSICS = ROOT / "gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json"


def configuration():
    pytest.importorskip("mjlab")
    from gear_sonic.envs.mjlab.sonic_true23_causal_multimotion_v14 import make_causal_multimotion_v14_env_cfg
    from gear_sonic.envs.mjlab.sonic_true23_native_model_actuation import apply_native_model_actuation_profile

    return apply_native_model_actuation_profile(
        make_causal_multimotion_v14_env_cfg(motion_file="unused.npz", num_envs=2, play=False),
        NativeModelActuationProfile.from_sim_config(PHYSICS),
    )


@pytest.fixture(scope="module")
def compiled():
    from mjlab.scene import Scene

    original = configuration()
    cfg = configure_nominal_scene(original, MODEL, PHYSICS)
    scene = Scene(cfg.scene, device="cpu")
    model = scene.compile()
    cfg.sim.mujoco.apply(model)
    return original, cfg, model


def test_constructed_scene_matches_referee_and_preserves_caller(compiled):
    original, cfg, model = compiled
    report = verify_nominal_scene(model, MODEL, PHYSICS)
    assert report["parameter_parity_passed"]
    assert report["collision_geom_count"] == 33
    assert not report["dance_tracking_qualified"]
    assert original.scene.terrain is not None and cfg.scene.terrain is None
    assert original.scene.entities["robot"].collisions
    assert not cfg.scene.entities["robot"].collisions
    assert original.actions["joint_pos"].profile == cfg.actions["joint_pos"].profile
    assert original.rewards == cfg.rewards and original.terminations == cfg.terminations


def test_complete_same_state_same_torque_cpu_step_parity(compiled):
    _, _, model = compiled
    _, reference, _ = prepare_true23_model(MODEL, PHYSICS)
    rng = np.random.default_rng(829)
    config = json.loads(PHYSICS.read_text())
    standing = config["initial_state"]
    q0 = np.r_[
        standing["base_position_m"], standing["base_quaternion_wxyz"], standing["joint_position_hardware_rad"]
    ]
    for index in range(10):
        a, b = mujoco.MjData(model), mujoco.MjData(reference)
        q = q0.copy()
        q[2] += rng.uniform(-0.01, 0.01)
        q[7:] += rng.uniform(-0.03, 0.03, 23)
        velocity, torque = rng.normal(0, 0.05, 29), rng.normal(0, 1, 23)
        a.qpos[:], b.qpos[:] = q, q
        a.qvel[:], b.qvel[:] = velocity, velocity
        a.ctrl[:], b.ctrl[:] = torque, torque
        for _ in range(10):
            mujoco.mj_step(model, a)
            mujoco.mj_step(reference, b)
        np.testing.assert_allclose(a.qpos, b.qpos, atol=1e-9, rtol=0, err_msg=str(index))
        np.testing.assert_allclose(a.qvel, b.qvel, atol=1e-8, rtol=0, err_msg=str(index))


def test_contract_hashes_every_mesh_and_does_not_claim_readiness():
    contract = nominal_scene_contract(MODEL, PHYSICS)
    assert len(contract["inputs"]) == 29
    assert not contract["hardware_authorized"] and not contract["deployment_ready"]


@pytest.mark.parametrize("changed", ("root", "friction", "solver", "effort", "derived_cache"))
def test_constructed_drift_is_rejected(compiled, changed):
    import copy

    model = copy.copy(compiled[2])
    if changed == "root":
        model.dof_armature[0] = 0.0
    elif changed == "friction":
        model.geom_friction[model.geom("floor").id, 0] = 0.6
    elif changed == "solver":
        model.opt.iterations = 10
    elif changed == "derived_cache":
        model.dof_M0[3] += 0.01
    else:
        model.jnt_actfrcrange[1, 1] += 1
    with pytest.raises(ValueError, match="differs from CPU referee"):
        verify_nominal_scene(model, MODEL, PHYSICS)


def test_rejects_wrong_motor_profile_and_competing_physics():
    cfg = configuration()
    cfg.actions["joint_pos"].profile = None
    with pytest.raises(ValueError, match="motor action profile"):
        configure_nominal_scene(cfg, MODEL, PHYSICS)
    cfg = configuration()
    cfg.scene.spec_fn = lambda spec: None
    with pytest.raises(ValueError, match="competing"):
        configure_nominal_scene(cfg, MODEL, PHYSICS)
