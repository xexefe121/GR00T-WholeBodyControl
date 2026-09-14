"""Physical contracts for the bounded offline shoulder experiment."""
import copy
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

DIAGNOSTICS = Path(__file__).resolve().parents[2] / 'artifacts/onboard_inspection_20260912'
sys.path.insert(0, str(DIAGNOSTICS))
import diagnose_pico_preview_failure as replay
from search_pico_shoulder_recovery import joint_index, rollout, seeds


@pytest.fixture
def plant():
    names = [f'joint_{i}' for i in range(23)]
    names[19] = 'right_shoulder_roll_joint'
    bodies = ''.join(f'<body pos="{i*.1} 0 0"><joint name="{name}" range="-1 1"/>'
                     '<geom type="sphere" size=".05" mass="1"/></body>'
                     for i, name in enumerate(names))
    motors = ''.join(f'<motor joint="{name}"/>' for name in names)
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><compiler angle="radian"/><option timestep=".002" gravity="0 0 0"/>'
        '<default><geom contype="0" conaffinity="0"/></default>'
        '<worldbody><body pos="0 0 1"><freejoint/><geom type="sphere" size=".1" mass="1"/>'
        f'{bodies}</body></worldbody><actuator>{motors}</actuator></mujoco>')
    contract = dict(kp=np.full(23, 20.), kd=np.ones(23), native_effort=np.full(23, .1),
                    native_velocity=np.full(23, 100.), joint_limits=np.tile([-1., 1.], (23, 1)))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, contract, data


def test_named_joint_and_legal_saturation_seeds(plant):
    model, contract, _ = plant
    assert joint_index(model, contract) == 19
    values = seeds(-1., 1., .2, -.3, 20., 1., .1, .4, -.5, .6)
    assert len(values) >= 257
    assert values.min() == -1. and values.max() == 1.
    for effort in (-.1, .1):
        assert .2 + (effort - .3) / 20. in values
    broken = dict(contract, joint_limits=contract['joint_limits'].copy())
    broken['joint_limits'][19, 0] = -.9
    with pytest.raises(AssertionError):
        joint_index(model, broken)


def test_delay_prefix_cannot_be_changed_by_replacement(plant):
    model, c, data = plant
    previous = np.full(23, .05)
    state = replay.integration_state(model, data)
    _, raw_a, applied_a = rollout(model, c, state, previous, np.ones(23), 6, steps=7)
    _, raw_b, applied_b = rollout(model, c, state, previous, -np.ones(23), 6, steps=7)
    np.testing.assert_array_equal(raw_a[:6], raw_b[:6])
    np.testing.assert_array_equal(applied_a[:6], applied_b[:6])
    assert not np.array_equal(applied_a[6], applied_b[6])
    assert np.any(abs(raw_a) > c['native_effort'])
    assert np.all(abs(applied_a) <= c['native_effort'])


def test_other_joint_failure_disqualifies_shoulder_candidate(plant):
    model, c, data = plant
    data.qpos[7 + 3] = 1.02
    result, _, _ = rollout(model, c, replay.integration_state(model, data),
                            np.zeros(23), np.zeros(23), 0, steps=1)
    assert not result['passed']
    assert 'native_joint_bound' in result['first_failure']['reasons']
    assert result['first_failure']['minimum_margin_joint'] == 'joint_3'


def test_full_state_roundtrip_preserves_warmstart_and_next_physics(plant):
    model, c, data = plant
    data.qacc_warmstart[:] = np.linspace(-.7, .9, model.nv)
    data.ctrl[:] = .04
    state = replay.integration_state(model, data)
    restored = replay.restore_physics(model, state)
    np.testing.assert_array_equal(restored.qacc_warmstart, data.qacc_warmstart)
    original = copy.copy(data)
    for _ in range(16):
        mujoco.mj_step(model, original)
        mujoco.mj_step(model, restored)
        np.testing.assert_array_equal(replay.integration_state(model, original),
                                      replay.integration_state(model, restored))


def test_replay_divergence_reports_component_and_releases_no_snapshot(plant, tmp_path, monkeypatch):
    model, c, data = plant
    states = [np.r_[data.qpos, data.qvel, data.time]]
    targets = np.full((2, 23), .1)
    torques = []
    for target in targets:
        replay.set_pd(data, target, c['kp'], c['kd'], c['native_effort'], c['joint_limits'])
        torques.append(data.ctrl.copy())
        mujoco.mj_step(model, data)
        states.append(np.r_[data.qpos, data.qvel, data.time])
    torques = np.asarray(torques)
    torques[1, 5] += .01
    run = tmp_path / 'input'; run.mkdir()
    output = tmp_path / 'verification'; output.mkdir()
    np.savez(run / 'trace.npz', states=states, targets=targets, torques=torques)
    (run / 'report.json').write_text(json.dumps(dict(clip='fixture',
        actual_pd_kp=c['kp'].tolist(), actual_pd_kd=c['kd'].tolist())))
    monkeypatch.setattr(replay, 'load_case', lambda clip: (model, c))
    with pytest.raises(RuntimeError, match='diverged'):
        replay.verify_run(run, [0], output)
    report = json.loads((output / 'report.json').read_text())
    assert report['first_difference']['physics_step'] == 1
    assert report['first_difference']['component'] == 'commanded_torque'
    assert report['first_difference']['index'] == 5
    assert not report['exact_applied_target_replay']
    assert not list(output.glob('boundary_*.npz'))
