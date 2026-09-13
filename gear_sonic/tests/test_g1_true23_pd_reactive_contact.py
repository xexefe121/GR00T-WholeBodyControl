from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.tests.test_g1_true23_contact_step_transition import native  # noqa: F401
from gear_sonic.utils.g1_true23_pd_reactive_contact import (
    TERMINAL_FIELDS,
    make_reactive_contact_law,
    terminal_nonregression,
)


@pytest.mark.parametrize("floor,expected", [(0.0, 0.011), (-0.002, 0.009)])
def test_state_shift_activates_absent_contact(native, monkeypatch, floor, expected):  # noqa: F811
    import gear_sonic.utils.g1_true23_pd_reactive_contact as reactive

    model, standing, _ = native

    class Query:
        def __init__(self, *args, **kwargs):
            pass

        def pose_rows(self, pose, columns):
            gradient = np.zeros(29)
            gradient[6] = 1.0
            return [dict(geoms=(7, 49), distance_m=float(pose[7]), joint_jacobian=gradient)]

    monkeypatch.setattr(reactive, "SelfCollisionLinearizer", Query)
    plant = SimpleNamespace(model=model, lower=np.full(23, -1.0), upper=np.ones(23))
    a, b = np.eye(81)[None], np.zeros((1, 81, 23))
    b[0, 6, 0] = 1.0
    poses = np.tile(standing, (2, 1))
    poses[:, 7] = 0.04  # Outside cached contact horizon at nominal state.
    stages = [dict(hessian=np.eye(23), gradient=np.zeros(23), gradient_state=np.zeros((23, 81)))]
    law, stats = make_reactive_contact_law(
        plant, dict(a=a, b=b), dict(qpos=poses), np.zeros((1, 23)), stages, 1.0, {(7, 49): floor}
    )
    np.testing.assert_array_equal(law(0, np.zeros(81)), np.zeros(23))
    shifted = np.zeros(81)
    shifted[6] = -0.05
    correction = law(0, shifted)
    assert correction[0] == pytest.approx(expected, abs=1e-12)
    assert stats["controls_with_active_contact_constraints"] == 1
    assert stats["controls_solved"] == 2


@pytest.mark.parametrize("field", TERMINAL_FIELDS)
def test_any_worsened_terminal_metric_is_rejected(field):
    initial = dict(lifecycle=dict.fromkeys(TERMINAL_FIELDS, 0.1))
    candidate = dict(lifecycle=dict.fromkeys(TERMINAL_FIELDS, 0.09))
    assert terminal_nonregression(initial, candidate)["accepted"]
    candidate["lifecycle"][field] = 0.10001
    result = terminal_nonregression(initial, candidate)
    assert not result["accepted"]
    assert result["violations"][0]["metric"] == field


def test_incomplete_or_nonfinite_terminal_evidence_is_not_accepted():
    initial = dict(lifecycle=dict.fromkeys(TERMINAL_FIELDS, 0.1))
    for value in (None, np.nan, np.inf):
        candidate = dict(lifecycle=dict.fromkeys(TERMINAL_FIELDS, value))
        assert not terminal_nonregression(initial, candidate)["accepted"]
