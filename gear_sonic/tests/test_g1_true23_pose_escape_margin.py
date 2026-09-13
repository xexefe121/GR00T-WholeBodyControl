from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from gear_sonic.scripts import diagnose_g1_true23_collision_pose_escape as diagnostic


def fake_problem():
    return SimpleNamespace(
        config=SimpleNamespace(maximum_root_rotation_l1_rad=0.45),
        layout=SimpleNamespace(dof_addresses=np.arange(6, 29)),
        evaluate=lambda x: (np.tile([0.999, 0, 0], 3), sparse.csc_matrix((9, 87)), None),
        qpos=lambda x: x,
    )


def test_solver_margin_tightens_only_optimization_and_cache_cannot_leak_into_final_audit(monkeypatch):
    group = {"frame": 0, "name": "test_norm", "indices": np.arange(3), "scales": np.ones(3), "radius": 1.0}
    monkeypatch.setattr(diagnostic, "residual_groups", lambda *args: ([group.copy()], 3))
    monkeypatch.setattr(
        diagnostic,
        "SelfCollisionLinearizer",
        lambda model: SimpleNamespace(
            pose_rows=lambda *args: [{"distance_m": 0.002, "joint_jacobian": np.zeros(23)}]
        ),
    )
    fit = diagnostic.PoseEscape(fake_problem(), None, None, interior_solver_margin=True)
    x = np.zeros(29)
    optimization = fit.evaluate(x)[0].copy()
    original = fit.evaluate(x, optimization=False)[0].copy()
    assert optimization[0] < 0 < original[0]
    assert optimization[-1] < original[-1]
    assert fit.groups[0]["radius"] == 1.0
    np.testing.assert_array_equal(fit.evaluate(x)[0], optimization)
    np.testing.assert_array_equal(fit.evaluate(x, optimization=False)[0], original)


@pytest.mark.parametrize("limit", [0, -0.1, 0.10001, np.nan, np.inf])
def test_pose_diagnostic_cannot_raise_upper_position_limit(monkeypatch, limit):
    monkeypatch.setattr(diagnostic, "residual_groups", lambda *args: ([], 3))
    with pytest.raises(ValueError, match="existing 10-cm"):
        diagnostic.PoseEscape(fake_problem(), None, None, upper_position_limit_m=limit)
