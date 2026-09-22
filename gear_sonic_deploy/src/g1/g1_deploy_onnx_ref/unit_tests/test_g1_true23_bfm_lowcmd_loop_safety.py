"""Static safety guard for the native G1 LowCmd path.

The loopback timing publisher remains available exactly as before.  The only
second construction path is deliberately explicit and must be preceded by all
real-endpoint arming and live-state pre-flight checks.
"""

from pathlib import Path
import re


def source() -> str:
    return (Path(__file__).parents[1] / "src" / "g1_true23_bfm_lowcmd_loop.cpp").read_text()


def _guard_containing(text: str, needle: str) -> str:
    position = text.index(needle)
    for match in re.finditer(r"if\s*\(([^{}]*)\)\s*\{", text):
        depth, cursor = 1, match.end()
        while depth and cursor < len(text):
            depth += (text[cursor] == "{") - (text[cursor] == "}")
            cursor += 1
        if match.end() <= position < cursor:
            return match.group(1)
    raise AssertionError(f"{needle!r} is not inside a braced if guard")


def test_loopback_publisher_and_sole_send_remain_present_and_guarded():
    text = source()
    assert "kSafeTopic" in text
    assert text.count("publisher->Write(MakeLowCmd(command, observed_mode_machine))") == 1
    send_guard = _guard_containing(text, "publisher->Write(MakeLowCmd(command, observed_mode_machine))")
    assert "args.dds_domain == kSafeDomain" in send_guard
    assert "IsLoopbackInterface(args.dds_interface)" in send_guard


def test_real_endpoint_publisher_requires_every_independent_arming_condition():
    text = source()
    preflight = text[text.index("RealArmRequestFailures"):text.index("LiveBringupPreflight")]
    for required in (
        "!args.arm", "!args.dds_domain_explicit", "!args.dds_interface_explicit",
        "args.operator_token_file.empty()", "args.operator_token.empty()", "age > 60",
        "token != args.operator_token",
    ):
        assert required in preflight
    live = text[text.index("LiveBringupPreflight"):text.index("void WriteProbeSummary")]
    for required in (
        "!observation.convertible", "observation.mode_machine != 4",
        "std::abs(observation.quaternion_norm - 1.0) > .01", "outside model limits",
        "20'000'000LL",
    ):
        assert required in live
    construction_guard = _guard_containing(text, "std::make_shared<unitree::robot::ChannelPublisher<LowCmd>>")
    assert "real_endpoint_armed" in construction_guard
    assert "real_endpoint_armed = true" in text
    assert "LiveBringupPreflight(args, estimator, failures)" in text


def test_hg_message_uses_mapped_slots_observed_machine_mode_and_crc():
    text = source()
    make_lowcmd = text[text.index("LowCmd MakeLowCmd"):text.index("double Percentile")]
    assert "result.mode_pr() = 0" in make_lowcmd
    assert "result.mode_machine() = observed_mode_machine" in make_lowcmd
    assert "for (auto& motor : result.motor_cmd())" in make_lowcmd
    assert "motor.mode() = 0" in make_lowcmd
    assert "motor_cmd().at(kHardwareSlots[i])" in make_lowcmd
    assert "motor.mode() = 1" in make_lowcmd
    assert "result.crc() = Crc32" in make_lowcmd


def test_native_loop_owns_ladder_and_all_abort_decisions():
    text = source()
    assert "class NativeBringupLadder" in text
    assert "kTargetMaxAgeNs = 100'000'000LL" in text
    for required in (
        "LowState is older than 20 ms", "policy target stale beyond 100 ms",
        "more than one consecutive deadline miss", "commanded joint outside model limits",
        # The brake check applies to the emitted command, not the raw policy
        # target: aborting on the raw target made the clamp unreachable at the
        # hand-over from the default pose (2026-09-20).
        "emitted command step exceeds brake bound", "measured joint position error exceeds limit",
        "measured joint velocity exceeds limit", "estimated tilt exceeds limit",
        "non-finite state or measured joint outside model limits", "operator liveness lost",
        "kAbortDampingRampNs", "kAbortZeroTorqueAfterNs", "ControlWire",
    ):
        assert required in text
    assert "command = ladder->Command" in text
    assert "last_operator_liveness_ns = MonotonicNs()" in text


def test_policy_stage_applies_the_brake_last():
    """The brake must be the final clamp, so nothing after it can exceed the step bound.

    Twice on 2026-09-20 a clamp applied after the brake - first the follow
    margin, then the model joint limits - moved the command further than the
    brake allows.  The step is also checked in double before the float store.
    """
    text = source()
    policy = text[text.index("kPolicy && target"):]
    policy = policy[:policy.index("output.kp[i] = static_cast<float>(operating_kp_[i]); output.kd[i]")]
    margin = policy.index("kCommandFollowMargin")
    limits = policy.index("next = std::clamp(next, limits_[i][0], limits_[i][1]);")
    brake = policy.index("next = std::clamp(next, previous - policy_step_, previous + policy_step_);")
    check = policy.index("std::abs(next - previous) > policy_step_ + 1e-9")
    store = policy.index("output.q[i] = static_cast<float>(next);")
    assert margin < limits < brake < check < store


def test_velocity_abort_needs_sustained_overspeed_but_keeps_its_limit():
    """A foot landing spikes an ankle for a tick; a runaway is sustained."""
    text = source()
    assert "kVelocityLimit" in text
    assert "inline constexpr int kVelocityExceededTicks = 10;" in text
    assert "++velocity_exceeded_ > kVelocityExceededTicks" in text
