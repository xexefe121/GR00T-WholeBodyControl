import json

import numpy as np
import pytest

from gear_sonic.scripts.train_g1_true23_generalist import MOTION_KEYS, validate_training_inputs


@pytest.fixture
def inputs(tmp_path):
    arrays = {key: np.zeros((20, 23), dtype=np.float32) for key in MOTION_KEYS}
    arrays.update(fps=np.array([50.0]))
    motion = tmp_path / "corpus.npz"
    np.savez(motion, **arrays)
    spans = tmp_path / "corpus.spans.json"
    payload = {
        "kind": "g1_true23_motion_corpus_spans_v1",
        "fps": 50,
        "clip_count": 1,
        "total_frames": 20,
        "spans": [{"start": 0, "length": 20}],
    }
    spans.write_text(json.dumps(payload))
    return motion, spans, payload


def test_regression_corpus_can_only_be_smoke(inputs):
    motion, spans, _ = inputs
    report = validate_training_inputs(motion, spans, None, smoke_only=True)
    assert report["smoke_only"] and not report["declared_recording_split_integrity_verified"]
    assert "recording_independence_verified" not in report
    assert not report["held_out_policy_generalization_verified"]
    with pytest.raises(ValueError, match="corpus-manifest"):
        validate_training_inputs(motion, spans, None, smoke_only=False)


@pytest.mark.parametrize("change", ["gap", "overlap", "short", "total", "fps"])
def test_span_rejection(inputs, change):
    motion, spans, payload = inputs
    if change in {"gap", "overlap"}:
        payload["spans"][0]["start"] = 1 if change == "gap" else -1
    elif change == "short":
        payload["spans"][0]["length"] = 10
    elif change == "total":
        payload["total_frames"] = 21
    else:
        payload["fps"] = 60
    spans.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        validate_training_inputs(motion, spans, None, smoke_only=True)


def test_nonfinite_motion_rejected(inputs):
    motion, spans, _ = inputs
    with np.load(motion) as archive:
        values = dict(archive)
    values["joint_pos"][0, 0] = np.nan
    np.savez(motion, **values)
    with pytest.raises(ValueError, match="invalid training payload"):
        validate_training_inputs(motion, spans, None, smoke_only=True)


def test_nominal_environment_has_exact_motor_contract():
    from gear_sonic.envs.mjlab.sonic_true23_causal_multimotion_v14 import make_causal_multimotion_v14_env_cfg
    from gear_sonic.envs.mjlab.sonic_true23_native_model_actuation import (
        NativeModelActuationActionCfg,
        apply_native_model_actuation_profile,
    )
    from gear_sonic.scripts.train_g1_true23_generalist import SIM_CONFIG
    from gear_sonic.utils.g1_true23_native_model_actuation import NativeModelActuationProfile

    cfg = make_causal_multimotion_v14_env_cfg(motion_file="unused.npz", num_envs=4)
    original_action = cfg.actions["joint_pos"]
    profile = NativeModelActuationProfile.from_sim_config(SIM_CONFIG)
    native = apply_native_model_actuation_profile(cfg, profile)
    assert cfg.actions["joint_pos"] is original_action
    assert isinstance(native.actions["joint_pos"], NativeModelActuationActionCfg)
    assert native.sim.mujoco.timestep == 0.002 and native.decimation == 10
    assert "stage_one_actuation_guard" not in native.terminations
    assert "invalid_native_model_actuation" in native.terminations
    assert len(native.scene.entities["robot"].articulation.actuators) == 23
    assert native.rewards["requested_effort_excess"].weight == -0.1
