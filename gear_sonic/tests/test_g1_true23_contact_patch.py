import copy
from pathlib import Path

import mujoco
import numpy as np
import pytest

from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE
from gear_sonic.utils.g1_true23_contact_patch import FrozenFloorSupportPatch
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256
from gear_sonic.utils.g1_true23_reference_support import floor_contact_map


@pytest.fixture
def floor_patch():
    root = Path(__file__).resolve().parents[2]
    model = mujoco.MjModel.from_xml_path(str(root / "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml"))
    model.geom_margin[:] = np.maximum(model.geom_margin, 0.002)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
    data = mujoco.MjData(model)
    data.qpos[7:] = SAFE_TARGET_DEFAULT_Q_HARDWARE
    data.qpos[2] = 0.8
    plane = model.geom("floor").id
    mujoco.mj_fwdPosition(model, data)
    lowest = min(
        mujoco.mj_geomDistance(model, data, plane, geom, 3.0, None)
        for geom in range(model.ngeom)
        if model.geom_bodyid[geom] and (model.geom_contype[geom] or model.geom_conaffinity[geom])
    )
    data.qpos[2] += 0.0004 - lowest
    mujoco.mj_fwdPosition(model, data)
    _, contacts = floor_contact_map(model, data, plane, 0.002)
    assert contacts
    return model, data, plane, contacts


def test_patch_recovers_surface_from_midpoint_and_does_not_mutate_source(floor_patch):
    model, data, plane, contacts = floor_patch
    model_hash, qpos = compiled_model_sha256(model), data.qpos.copy()
    patch = FrozenFloorSupportPatch(model, data, contacts, plane)
    values, _ = patch.evaluate(qpos)
    np.testing.assert_allclose(values, 0.00195 - np.array([row["distance_m"] for row in contacts]), atol=1e-12)
    raised = qpos.copy()
    raised[2] += 0.01
    lifted, _ = patch.evaluate(raised)
    np.testing.assert_allclose(lifted, values - 0.01, atol=1e-12)
    assert np.all(lifted < 0)
    assert compiled_model_sha256(model) == model_hash
    np.testing.assert_array_equal(data.qpos, qpos)


def test_patch_all_26_analytical_derivatives_match_central_differences(floor_patch):
    model, data, plane, contacts = floor_patch
    patch = FrozenFloorSupportPatch(model, data, contacts, plane)
    qpos = data.qpos.copy()
    _, analytical = patch.evaluate(qpos)
    for column, index in enumerate(np.r_[0:3, 7:30]):
        delta = np.zeros(30)
        delta[index] = 1e-6
        plus, _ = patch.evaluate(qpos + delta, jacobian=False)
        minus, _ = patch.evaluate(qpos - delta, jacobian=False)
        np.testing.assert_allclose(analytical[:, column], (plus - minus) / 2e-6, atol=2e-10, rtol=1e-7)


@pytest.mark.parametrize("guard", [0, -1e-4, 0.002, np.nan])
def test_patch_cannot_expand_candidate_contact_band(floor_patch, guard):
    model, data, plane, contacts = floor_patch
    with pytest.raises(ValueError, match="guard < gap"):
        FrozenFloorSupportPatch(model, data, contacts, plane, guard_m=guard)


def test_patch_rejects_forged_midpoints_and_handles_empty_support(floor_patch):
    model, data, plane, contacts = floor_patch
    bad = copy.deepcopy(contacts)
    bad[0]["position_w"][2] += 0.001
    with pytest.raises(ValueError, match="midpoint/distance"):
        FrozenFloorSupportPatch(model, data, bad, plane)
    patch = FrozenFloorSupportPatch(model, data, [], plane)
    values, derivative = patch.evaluate(data.qpos)
    assert values.shape == (0,) and derivative.shape == (0, 26)
