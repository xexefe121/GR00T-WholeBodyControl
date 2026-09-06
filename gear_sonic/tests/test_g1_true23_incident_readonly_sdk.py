"""Real HG/API IDL and native CRC, with every DDS entry point forbidden.

Run with the pinned Unitree SDK and Cyclone DDS Python dependency installed.
No participant, publisher, reader, socket or robot is created by these tests.
"""

import base64
import gzip
import json

import pytest

pytest.importorskip("unitree_sdk2py", reason="offline SDK codec tests need unitree_sdk2_python")

from unitree_sdk2py.core import channel  # noqa: E402
from unitree_sdk2py.idl.default import (  # noqa: E402
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__LowState_,
)
from unitree_sdk2py.idl.unitree_api.msg.dds_ import (  # noqa: E402
    Request_,
    RequestHeader_,
    RequestIdentity_,
    RequestLease_,
    RequestPolicy_,
    Response_,
    ResponseHeader_,
    ResponseStatus_,
)

from gear_sonic.scripts import record_g1_true23_incident_readonly as recorder  # noqa: E402
from gear_sonic.utils.g1_true23_incident_capture import PacketCapture, TOPICS  # noqa: E402


@pytest.fixture(autouse=True)
def forbid_dds_and_sockets(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("offline test attempted DDS or network operation")

    for name in ("Domain", "DomainParticipant", "DataWriter", "DataReader", "Topic"):
        monkeypatch.setattr(channel, name, forbidden)
    monkeypatch.setattr(recorder.socket, "socket", forbidden)


@pytest.fixture
def api():
    return recorder.load_readonly_api()


def decode(api, topic, message):
    return recorder.decode_packet(topic, message.serialize(), api[2], api[3])


def test_lowstate_real_35_slot_idl_crc_and_complete_raw_payload(api, tmp_path):
    state = unitree_hg_msg_dds__LowState_()
    state.tick = (1 << 32) - 1
    state.mode_machine = 4
    state.version[:] = [123, 456]
    state.imu_state.quaternion[:] = [1.0, 0.0, 0.0, 0.0]
    state.wireless_remote[:] = list(range(40))
    for slot, motor in enumerate(state.motor_state):
        motor.mode = slot % 2
        motor.q = slot / 4
        motor.dq = -slot / 4
        motor.ddq = slot / 8
        motor.tau_est = slot / 2
        motor.motorstate = (1 << 30) | slot
        motor.vol = 48.0
        motor.temperature[:] = [slot, -slot]
        motor.sensor[:] = [slot, 100 + slot]
        motor.reserve[:] = [slot] * 4
    state.crc = api[3].Crc(state)
    raw = state.serialize()
    decoded = decode(api, "rt/lowstate", state)
    assert decoded["crc_matches"] is True
    assert decoded["version_raw"] == [123, 456]
    assert decoded["tick"] == (1 << 32) - 1
    assert decoded["motor_status"] == [(1 << 30) | slot for slot in range(35)]
    assert decoded["q_rad"] == [slot / 4 for slot in range(35)]
    assert decoded["motor_temperature_raw"][34] == [34, -34]
    assert decoded["wireless_remote_hex"] == bytes(range(40)).hex()
    output = tmp_path / "real-codec"
    capture = PacketCapture(
        output, metadata={}, decode=lambda topic, payload: recorder.decode_packet(topic, payload, api[2], api[3])
    )
    capture.receive("rt/lowstate", state)
    report = capture.finish()
    assert report["all_observed_callbacks_preserved"] is True
    with gzip.open(output / "packets.jsonl.gz", "rt") as stream:
        packet = [json.loads(line) for line in stream][1]
    assert base64.b64decode(packet["reserialized_cdr_b64"]) == raw
    restored = api[2]["LowState_"].deserialize(raw)
    assert restored.motor_state[34].sensor == [34, 134]
    assert restored.motor_state[34].ddq == 34 / 8
    assert restored.motor_state[34].reserve == [34] * 4
    state.motor_state[0].q = 1.0  # CRC deliberately remains from previous state.
    assert decode(api, "rt/lowstate", state)["crc_matches"] is False


@pytest.mark.parametrize("topic", ["rt/lowcmd", "rt/user_lowcmd"])
def test_real_command_decode_never_claims_received_or_applied(api, topic):
    command = unitree_hg_msg_dds__LowCmd_()
    command.mode_machine = 4
    command.motor_cmd[26].q = 0.25
    command.motor_cmd[26].kp = 20.0
    command.motor_cmd[26].kd = 2.0
    command.motor_cmd[26].tau = 1.0
    command.crc = api[3].Crc(command)
    decoded = decode(api, topic, command)
    assert decoded["q_rad"][26] == 0.25
    assert decoded["crc_received"] == decoded["crc_computed"]
    assert decoded["robot_received_or_applied_this_command_proven"] is False


@pytest.mark.parametrize("service", ["sport", "motion_switcher"])
def test_real_rpc_identity_status_and_payload_preserved_without_causality(api, service):
    # This SDK revision's default RequestLease factory has the wrong arity.
    # Use the real IDL constructors, as its RPC client does; do not alter SDK.
    request = Request_(
        RequestHeader_(RequestIdentity_(0, 0), RequestLease_(0), RequestPolicy_(0, False)),
        "",
        [],
    )
    request.header.identity.id = 123456789
    request.header.identity.api_id = 7110
    request.parameter = '{"fsm_id": 801}'
    decoded = decode(api, f"rt/api/{service}/request", request)
    assert decoded["request_id"] == 123456789
    assert decoded["api_id"] == 7110
    assert decoded["parameter"] == request.parameter
    response = Response_(ResponseHeader_(RequestIdentity_(0, 0), ResponseStatus_(0)), "", [])
    response.header.identity.id = request.header.identity.id
    response.header.identity.api_id = request.header.identity.api_id
    response.header.status.code = 3104
    response.data = "injected rejection"
    result = decode(api, f"rt/api/{service}/response", response)
    assert result["status_code"] == 3104
    assert result["data"] == "injected rejection"
    assert result["rpc_match_or_causal_relation_proven"] is False


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_real_nonfinite_idl_state_is_saved_not_discarded(api, tmp_path, value):
    state = unitree_hg_msg_dds__LowState_()
    state.motor_state[0].q = value
    state.crc = api[3].Crc(state)
    output = tmp_path / "nonfinite-idl"
    capture = PacketCapture(
        output, metadata={}, decode=lambda topic, payload: recorder.decode_packet(topic, payload, api[2], api[3])
    )
    capture.receive("rt/lowstate", state)
    assert capture.finish()["all_observed_callbacks_preserved"] is True
    with gzip.open(output / "packets.jsonl.gz", "rt") as stream:
        packet = [json.loads(line) for line in stream][1]
    assert packet["decoded"]["q_rad"][0] == {"nonfinite_float": repr(value)}


def fake_api(real, calls, *, fail_on=None, close_error=False):
    def initialize(domain, interface):
        calls.append(("initialize", domain, interface))
        if fail_on == "initialize":
            raise RuntimeError("injected initialization failure")

    class Subscriber:
        def __init__(self, topic, message_type):
            self.topic = topic
            calls.append(("subscriber", topic, message_type.__name__))

        def Init(self, callback, queue_len):
            calls.append(("init", self.topic, queue_len))
            if fail_on == self.topic:
                raise RuntimeError("injected subscription failure")
            if self.topic == "rt/lowstate":
                state = unitree_hg_msg_dds__LowState_()
                state.tick = 20
                state.mode_machine = 4
                state.crc = real[3].Crc(state)
                callback(state)

        def Close(self):
            calls.append(("close", self.topic))
            if close_error:
                raise RuntimeError("injected close failure")

    return initialize, Subscriber, real[2], real[3]


def test_collect_uses_exact_seven_subscriptions_no_hidden_sdk_queue(api, tmp_path):
    calls = []
    report = recorder.collect(
        tmp_path / "fake-transport", interface="fake0", duration_s=0.1, api=fake_api(api, calls)
    )
    assert calls[0] == ("initialize", 0, "fake0")
    assert [entry[1] for entry in calls if entry[0] == "subscriber"] == list(TOPICS)
    assert [entry[2] for entry in calls if entry[0] == "init"] == [0] * 7
    assert [entry[1] for entry in calls if entry[0] == "close"] == list(reversed(TOPICS))
    assert report["capture_completed"] is True
    assert report["dds_subscriptions_opened"] is True  # Fake transport only.
    assert report["observed_lowstate"] is True
    assert report["robot_commands_published"] is False
    with gzip.open(tmp_path / "fake-transport" / "packets.jsonl.gz", "rt") as stream:
        metadata = json.loads(next(stream))["metadata"]
    assert metadata["schema_source_sha256"]
    assert any(path.endswith("_LowState_.py") for path in metadata["schema_source_sha256"])
    assert any("crc" in path for path in metadata["schema_source_sha256"])


@pytest.mark.parametrize("fail_on", ["initialize", "rt/lowstate", "rt/user_lowcmd"])
def test_failed_setup_closes_every_created_subscription_and_retains_failure(api, tmp_path, fail_on):
    calls = []
    report = recorder.collect(
        tmp_path / "failed",
        interface="fake0",
        duration_s=0.1,
        api=fake_api(api, calls, fail_on=fail_on, close_error=True),
    )
    created = [entry[1] for entry in calls if entry[0] == "subscriber"]
    closed = [entry[1] for entry in calls if entry[0] == "close"]
    assert closed == list(reversed(created))
    assert report["capture_completed"] is False
    assert (
        "injected initialization failure" in report["transport_error"]
        if fail_on == "initialize"
        else "injected subscription failure" in report["transport_error"]
    )
    if created:
        assert "injected close failure" in report["transport_error"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"duration_s": 0},
        {"duration_s": 301},
        {"duration_s": float("nan")},
        {"interface": ""},
        {"multicast_local_ip": "not-an-IP"},
    ],
)
def test_invalid_cli_inputs_fail_before_sdk_or_evidence(tmp_path, kwargs, monkeypatch):
    monkeypatch.setattr(recorder, "load_readonly_api", lambda: pytest.fail("SDK loaded before validation"))
    arguments = {"interface": "fake0", "duration_s": 0.1, **kwargs}
    with pytest.raises(ValueError):
        recorder.collect(tmp_path / "absent", **arguments)
    assert not (tmp_path / "absent").exists()


def test_existing_output_rejected_before_sdk_load(tmp_path, monkeypatch):
    monkeypatch.setattr(recorder, "load_readonly_api", lambda: pytest.fail("SDK loaded before validation"))
    with pytest.raises(FileExistsError):
        recorder.collect(tmp_path, interface="fake0", duration_s=0.1)


def test_changed_recorder_source_is_reported_not_qualified(api, tmp_path, monkeypatch):
    real_hash = recorder.sha256
    counter = {}

    def changing_hash(path):
        key = str(path)
        counter[key] = counter.get(key, 0) + 1
        if key.endswith("record_g1_true23_incident_readonly.py") and counter[key] > 1:
            return "0" * 64
        return real_hash(path)

    monkeypatch.setattr(recorder, "sha256", changing_hash)
    report = recorder.collect(
        tmp_path / "changed-source", interface="fake0", duration_s=0.1, api=fake_api(api, [])
    )
    assert report["capture_completed"] is False
    assert "capture_source_changed" in report["transport_error"]


@pytest.mark.parametrize(
    "flag", [None, "capture_completed", "all_observed_callbacks_preserved", "observed_lowstate"]
)
def test_cli_exit_is_capture_completeness_not_hardware_readiness(flag, monkeypatch):
    report = {"capture_completed": True, "all_observed_callbacks_preserved": True, "observed_lowstate": True}
    if flag:
        report[flag] = False
    monkeypatch.setattr(recorder, "collect", lambda *args, **kwargs: report)
    monkeypatch.setattr(recorder.sys, "argv", ["capture", "--interface", "fake0", "--output-directory", "unused"])
    assert recorder.main() == (0 if flag is None else 2)
