import copy
from types import SimpleNamespace

import pytest
import torch
from mjlab.utils.noise import UniformNoiseCfg

from gear_sonic.utils.g1_true23_sensor_noise_scale import EXPECTED_TERMS, ScaledUniformNoiseCfg, scale_sensor_noise


def config():
    cfg = SimpleNamespace(
        observations={
            name: SimpleNamespace(enable_corruption=True, terms={}) for name in ("tokenizer", "policy", "critic")
        }
    )
    for (group, term), (low, high) in EXPECTED_TERMS.items():
        cfg.observations[group].terms[term] = SimpleNamespace(noise=UniformNoiseCfg(n_min=low, n_max=high))
    cfg.observations["policy"].terms["joint_pos_rel"] = SimpleNamespace(noise=None)
    return cfg


@pytest.mark.parametrize("scale", [0.0, 0.25, 1.0])
def test_random_draws_preserved_at_every_amplitude(scale):
    state = torch.random.get_rng_state()
    try:
        data = torch.arange(48, dtype=torch.float32).reshape(8, 6) / 32
        torch.manual_seed(913)
        original = UniformNoiseCfg(n_min=-0.05, n_max=0.05).apply(data)
        rng = torch.random.get_rng_state()
        torch.manual_seed(913)
        actual = ScaledUniformNoiseCfg(n_min=-0.05, n_max=0.05, amplitude_scale=scale).apply(data)
        assert torch.equal(torch.random.get_rng_state(), rng)
        expected = original if scale == 1 else data if scale == 0 else data + scale * (original - data)
        assert torch.equal(actual, expected)
    finally:
        torch.random.set_rng_state(state)


@pytest.mark.parametrize("scale", [True, -1, 1.1, float("nan"), float("inf"), "0"])
def test_invalid_amplitude_rejected_without_partial_changes(scale):
    cfg = config()
    original = copy.deepcopy(cfg)
    with pytest.raises(ValueError):
        scale_sensor_noise(cfg, scale)
    assert cfg == original


def test_exact_terms_only_critic_and_corruption_unchanged():
    cfg = config()
    critic = cfg.observations["critic"]
    report = scale_sensor_noise(cfg, 0)
    assert cfg.observations["critic"] is critic
    assert cfg.observations["policy"].terms["joint_pos_rel"].noise is None
    assert all(group.enable_corruption for group in cfg.observations.values())
    assert report["zero_scale_still_consumes_original_random_draws"] is True
    assert report["deployment_ready"] is False
    for group, term in EXPECTED_TERMS:
        assert cfg.observations[group].terms[term].noise.amplitude_scale == 0


@pytest.mark.parametrize("change", ["missing", "extra", "bounds", "operation", "cached", "disabled"])
def test_changed_sensor_contract_rejected_atomically(change):
    cfg = config()
    group = cfg.observations["policy"]
    noise = group.terms["base_ang_vel"].noise
    if change == "missing":
        del group.terms["base_ang_vel"]
    elif change == "extra":
        group.terms["unrecognized"] = SimpleNamespace(noise=UniformNoiseCfg())
    elif change == "bounds":
        noise.n_min = -0.1
    elif change == "operation":
        noise.operation = "scale"
    elif change == "cached":
        noise.apply(torch.ones(1))
    else:
        group.enable_corruption = False
    with pytest.raises(ValueError):
        scale_sensor_noise(cfg, 0.25)
    assert type(cfg.observations["tokenizer"].terms["motion_anchor_ori_b"].noise) is UniformNoiseCfg


def test_probe_cannot_be_scaled_twice():
    cfg = config()
    scale_sensor_noise(cfg, 0.25)
    with pytest.raises(ValueError):
        scale_sensor_noise(cfg, 1)
