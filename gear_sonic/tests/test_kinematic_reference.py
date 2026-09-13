import copy
from pathlib import Path

import numpy as np
import pytest

from gear_sonic.teleop.kinematic_reference import PreparedNative23Reference
from gear_sonic.scripts.replay_g1_true23_pico_packets_zmq import load_reference_packets
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import CleanTrue23MujocoController, NATIVE_TO_MJ
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS


@pytest.fixture
def controller():
    root = Path(__file__).resolve().parents[2].parent / "GR00T-WholeBodyControl"
    if not (root / MODEL).is_file():
        pytest.skip("original native23 assets unavailable")
    return CleanTrue23MujocoController(model_path=root / MODEL, physics_path=root / PHYSICS, policy=None)


@pytest.fixture
def packet():
    path = Path("/mnt/c/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/walk002/causal_packets.json")
    if not path.is_file():
        pytest.skip("saved PICO-derived reference unavailable")
    return load_reference_packets(path)[0]


def test_exact_reference_and_no_plant_mutation(controller, packet):
    prepared = PreparedNative23Reference(controller)
    before = {
        key: getattr(controller.data, key).copy() for key in ("qpos", "qvel", "qacc", "ctrl", "qacc_warmstart")
    }
    original_packet = copy.deepcopy(packet)
    expected = controller.retarget_pico_reference_packet(packet)
    actual = prepared.retarget(packet)
    assert actual == expected
    assert packet == original_packet
    assert prepared.height_data is not controller.data and prepared.reference_data is not controller.data
    for key, value in before.items():
        np.testing.assert_array_equal(getattr(controller.data, key), value)
    assert controller.completed == 0 and controller.data.time == 0
    actual["causal_history_lower_body"][0] += 1
    assert packet == original_packet


def test_height_reuse_ignores_stale_dynamic_scratch(controller, packet):
    prepared = PreparedNative23Reference(controller)
    q = np.asarray(packet["q_ref23_native"])[NATIVE_TO_MJ]
    expected = controller.reference_root_height(q)
    prepared.height_data.qvel[:] = 27
    prepared.height_data.ctrl[:] = -15
    prepared.height_data.qacc_warmstart[:] = 83
    assert prepared.root_height(q) == expected
    assert prepared.root_height(q + 0.001) == controller.reference_root_height(q + 0.001)
    assert prepared.root_height(q) == expected


@pytest.mark.parametrize("mutation", ["nan", "keys", "proof", "time"])
def test_invalid_packet_preserves_rejection(controller, packet, mutation):
    prepared = PreparedNative23Reference(controller)
    bad = copy.deepcopy(packet)
    if mutation == "nan":
        bad["q_ref23_native"][0] = float("nan")
    elif mutation == "keys":
        bad["extra"] = True
    elif mutation == "proof":
        bad["qd_ref23_native"][0] += 1
    else:
        bad["control_monotonic_ns"] += 1
    for fn in (prepared.retarget, controller.retarget_pico_reference_packet):
        with pytest.raises(ValueError):
            fn(bad)
    assert controller.completed == 0 and controller.data.time == 0


def test_install_only_fresh_owned_instance(controller):
    prepared = PreparedNative23Reference(controller)
    prepared.install(controller)
    assert controller.retarget_pico_reference_packet.__self__ is prepared
    controller.completed = 1
    with pytest.raises(ValueError, match="before"):
        prepared.install(controller)
    with pytest.raises(ValueError, match="before"):
        PreparedNative23Reference(controller)
