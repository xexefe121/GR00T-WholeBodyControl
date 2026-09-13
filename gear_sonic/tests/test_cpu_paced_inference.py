from types import SimpleNamespace

import onnxruntime as ort
import pytest

from gear_sonic.teleop import cpu_paced_inference as runtime
from gear_sonic.scripts.run_g1_true23_prepared_paced_sim import full_recorded_sim_success


def test_only_spinning_changes_from_ort_defaults():
    options, defaults = runtime.nonspinning_options(), ort.SessionOptions()
    for name in ("intra_op_num_threads", "inter_op_num_threads", "execution_mode", "graph_optimization_level"):
        assert getattr(options, name) == getattr(defaults, name)
    for name in ("session.intra_op.allow_spinning", "session.inter_op.allow_spinning"):
        assert options.get_session_config_entry(name) == "0"


def test_partial_source_is_not_cli_success_even_when_balance_completes():
    result = dict(
        physical_screen_passed=True, measured_compute_deadlines_passed=True, full_source_sonic_completed=False
    )
    assert not full_recorded_sim_success(result)
    result["full_source_sonic_completed"] = True
    assert full_recorded_sim_success(result)
    for name in result:
        assert not full_recorded_sim_success({**result, name: False})
    assert not full_recorded_sim_success({})


def test_failed_fallback_build_leaves_original_policies_installed(monkeypatch):
    controller = SimpleNamespace(
        completed=0, data=SimpleNamespace(time=0), fallback_active=False, policy=object(), fallback_policy=object()
    )
    old = controller.policy, controller.fallback_policy
    monkeypatch.setattr(runtime, "ExactHashSonicPolicy", lambda **kwargs: object())

    def fail(*args, **kwargs):
        raise ValueError("bad fallback hash")

    monkeypatch.setattr(runtime, "UnitreeZeroVelocityFallbackPolicy", fail)
    identity = dict(
        diagnostic_pair=dict(encoder=dict(path="encoder"), decoder=dict(path="decoder")),
        encoder_sha256="a",
        decoder_sha256="b",
    )
    with pytest.raises(ValueError, match="bad fallback hash"):
        runtime.prepare_nonspinning_sessions(controller, identity, "/unused")
    assert (controller.policy, controller.fallback_policy) == old


@pytest.mark.parametrize("completed,time,fallback", [(1, 0, False), (0, 0.002, False), (0, 0, True)])
def test_not_installable_after_physics_or_fallback(completed, time, fallback):
    controller = SimpleNamespace(completed=completed, data=SimpleNamespace(time=time), fallback_active=fallback)
    with pytest.raises(ValueError, match="before"):
        runtime.prepare_nonspinning_sessions(controller, {}, "/unused")
