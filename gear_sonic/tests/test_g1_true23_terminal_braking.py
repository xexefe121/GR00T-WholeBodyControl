from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from gear_sonic.utils.g1_true23_terminal_braking import (
    TerminalBrakingTaskExtension,
    audit_terminal_braking,
    terminal_braking_rows,
)


@pytest.mark.parametrize("sign", [-1, 1])
def test_terminal_joint_inside_bounds_can_still_have_no_safe_next_sample(sign):
    q = np.zeros((5, 23))
    q[-2, 3], q[-1, 3] = sign * 0.94, sign * 0.99
    proof = audit_terminal_braking(q, np.full(23, -1.0), np.ones(23))
    assert not proof["passed"]
    assert proof["failed_rows"][0]["joint_index"] == 3
    assert proof["maximum_joint_position_violation_rad"] == pytest.approx(0.008)
    assert not proof["deployment_ready"]


def test_multistep_braking_not_just_the_first_future_sample_is_checked():
    q = np.zeros((4, 23))
    q[-2, 0], q[-1, 0] = 0.82, 0.92
    proof = audit_terminal_braking(q, np.full(23, -1.0), np.ones(23))
    assert not proof["passed"]
    assert {r["braking_step"] for r in proof["failed_rows"]} == {2, 3}
    assert proof["maximum_joint_position_violation_rad"] == pytest.approx(0.028)


def test_braking_rows_equal_direct_discrete_maximum_deceleration_envelope():
    rng = np.random.default_rng(19)
    q = rng.uniform(-0.5, 0.5, (10, 23))
    matrix, lower, upper = terminal_braking_rows(10, np.full(23, -1.0), np.ones(23))
    x = np.column_stack((np.zeros((10, 6)), q))
    actual = (matrix @ x.ravel()).reshape(4, 23)
    for k in range(1, 5):
        np.testing.assert_allclose(actual[k - 1], q[-1] + k * (q[-1] - q[-2]))
        np.testing.assert_allclose(lower.reshape(4, 23)[k - 1], -1 - 0.032 * k * (k + 1) / 2)
        np.testing.assert_allclose(upper.reshape(4, 23)[k - 1], 1 + 0.032 * k * (k + 1) / 2)
    assert not matrix[:, : 8 * 29].nnz


def test_task_extension_keeps_original_rows_and_exact_jacobian():
    x = np.zeros((5, 29))
    original_jac = sparse.csc_matrix(np.ones((3, x.size)))
    original = SimpleNamespace(
        original_times=np.arange(5),
        groups=[{"name": "original"}],
        evaluate=lambda value, derivatives=True: (
            original_jac @ value.ravel(),
            original_jac if derivatives else None,
        ),
    )
    extension = TerminalBrakingTaskExtension(original, x, np.full(23, -1.0), np.ones(23))
    residual, jacobian = extension.evaluate(x)
    assert extension.groups[0] == original.groups[0]
    assert len(extension.groups) == 93
    np.testing.assert_array_equal(jacobian[:3].toarray(), original_jac.toarray())
    perturb = x.copy()
    perturb[-1, 10] += 1e-6
    changed, _ = extension.evaluate(perturb, derivatives=False)
    np.testing.assert_allclose((changed - residual) / 1e-6, jacobian[:, 4 * 29 + 10].toarray().ravel(), atol=1e-9)
    assert extension.evaluate(x, derivatives=False)[1] is None


def test_stopped_terminal_reference_passes_and_invalid_values_rejected():
    q = np.zeros((3, 23))
    assert audit_terminal_braking(q, np.full(23, -1.0), np.ones(23))["passed"]
    q[-1, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        audit_terminal_braking(q, np.full(23, -1.0), np.ones(23))
    with pytest.raises(ValueError, match="positive"):
        terminal_braking_rows(3, np.full(23, -1.0), np.ones(23), dt=0)
