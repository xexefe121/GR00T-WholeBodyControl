import numpy as np
import pytest

from gear_sonic.tests.test_g1_true23_virtual_source_history import predictor as predictor
from gear_sonic.utils.g1_true23_momentum_source_model import (
    MISSING_V,
    OBSERVED_V,
    MissingMomentumAssimilation,
    MomentumSourceModel,
)
from gear_sonic.utils.g1_true23_virtual_source_model import KEEP_HW


def native_state(model):
    return np.r_[model.data.qpos[:7], model.data.qpos[7 + KEEP_HW]], np.r_[
        model.data.qvel[:6], model.data.qvel[6 + KEEP_HW]
    ]


def test_exact_identity_skips_mass_recompute(predictor, monkeypatch):
    assimilation = MissingMomentumAssimilation(predictor.model)
    q, v = native_state(predictor)
    monkeypatch.setattr(assimilation, "mass", lambda _: pytest.fail("identity should not recompute inertia"))
    result, record = assimilation.proposal(predictor.data.qpos.copy(), predictor.data.qvel.copy(), q, v)
    np.testing.assert_array_equal(result, predictor.data.qvel[MISSING_V])
    assert record["exact_identity"] is True


@pytest.mark.parametrize("change", ["velocity", "configuration", "both"])
def test_missing_momentum_equation_and_no_state_mutation(predictor, change):
    assimilation = MissingMomentumAssimilation(predictor.model)
    old_q = predictor.data.qpos.copy()
    old_v = np.linspace(-0.15, 0.15, 35)
    q = np.r_[old_q[:7], old_q[7 + KEEP_HW]]
    v = np.r_[old_v[:6], old_v[6 + KEEP_HW]]
    if change in ("velocity", "both"):
        v[0] += 0.2
        v[6:] += 0.15
    if change in ("configuration", "both"):
        q[7:] += 0.1
    for array in (q, v, old_q, old_v):
        array.setflags(write=False)
    actual_q, actual_v = predictor.data.qpos.copy(), predictor.data.qvel.copy()
    missing_v, record = assimilation.proposal(old_q, old_v, q, v)
    new_q, new_v = old_q.copy(), old_v.copy()
    new_q[:7], new_q[7 + KEEP_HW] = q[:7], q[7:]
    new_v[:6], new_v[6 + KEEP_HW] = v[:6], v[6:]
    new_v[MISSING_V] = missing_v
    np.testing.assert_allclose(
        assimilation.mass(new_q)[MISSING_V] @ new_v,
        assimilation.mass(old_q)[MISSING_V] @ old_v,
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_array_equal(new_v[OBSERVED_V], v)
    np.testing.assert_array_equal(predictor.data.qpos, actual_q)
    np.testing.assert_array_equal(predictor.data.qvel, actual_v)
    assert np.max(np.abs(record["missing_velocity_correction"])) > 1e-6
    assert record["conditional_correction_energy_j"] >= 0
    assert not np.shares_memory(missing_v, old_v)


def momentum_model(predictor):
    return MomentumSourceModel(predictor.path, predictor.parameters, source_effort29=predictor.effort)


def test_twenty_consistent_source_controls_exact_identity(predictor):
    corrected = momentum_model(predictor)
    q, v = native_state(predictor)
    q[2] = 1.1
    for i in range(20):
        raw = np.full(29, np.sin(i * 0.1) * 0.01, np.float32)
        before_q, before_v = q.copy(), v.copy()
        expected = predictor.advance(q, v, raw)
        actual = corrected.advance(q, v, raw)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(q, before_q)
        np.testing.assert_array_equal(v, before_v)
        np.testing.assert_array_equal(corrected.data.qpos, predictor.data.qpos)
        np.testing.assert_array_equal(corrected.data.qvel, predictor.data.qvel)
        q, v = native_state(predictor)
    assert corrected.descriptor()["maximum_missing_velocity_correction_rad_s"] == 0
    assert corrected.assimilation_arrays()["exact_identity"].all()


@pytest.mark.parametrize("bad", ["q_nonfinite", "v_nonfinite", "raw_nonfinite", "bad_quat"])
def test_invalid_input_terminal_before_state_write(predictor, bad):
    model = momentum_model(predictor)
    q, v = native_state(model)
    raw = np.zeros(29, np.float32)
    if bad == "q_nonfinite":
        q[0] = np.nan
    elif bad == "v_nonfinite":
        v[0] = np.inf
    elif bad == "raw_nonfinite":
        raw[0] = np.nan
    else:
        q[3:7] = 0
    before_q, before_v = model.data.qpos.copy(), model.data.qvel.copy()
    with pytest.raises(ValueError):
        model.advance(q, v, raw)
    assert model.failed and model.completed == 0
    np.testing.assert_array_equal(model.data.qpos, before_q)
    np.testing.assert_array_equal(model.data.qvel, before_v)
    q, v = native_state(model)
    with pytest.raises(RuntimeError):
        model.advance(q, v, np.zeros(29, np.float32))
