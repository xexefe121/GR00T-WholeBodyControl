"""Static safety guard for the Fix 5 timing loop.

This test deliberately checks the loop source rather than a runtime flag: a
future edit cannot introduce a production low-command publish topic unnoticed.
"""

from pathlib import Path
import re


def test_fix5_loop_has_only_the_compiled_test_lowcmd_publish_topic():
    source = (Path(__file__).parents[1] / "src" / "g1_true23_bfm_lowcmd_loop.cpp").read_text()
    published_topics = re.findall(r"ChannelPublisher<LowCmd>>\(std::string\(([^)]*)\)\)", source)
    assert published_topics == ["kSafeTopic"]
    assert '"rt/lowcmd"' not in source
    assert '"eth0"' not in source


def _guard_containing(source: str, needle: str) -> str:
    """Return the condition of the braced if-block containing ``needle``."""
    position = source.index(needle)
    for match in re.finditer(r"if\s*\(([^{}]*)\)\s*\{", source):
        depth = 1
        cursor = match.end()
        while depth and cursor < len(source):
            if source[cursor] == "{":
                depth += 1
            elif source[cursor] == "}":
                depth -= 1
            cursor += 1
        if match.end() <= position < cursor:
            return match.group(1)
    raise AssertionError(f"{needle!r} is not inside a braced if guard")


def test_lowcmd_publisher_construction_and_send_are_both_domain_and_loopback_guarded():
    source = (Path(__file__).parents[1] / "src" / "g1_true23_bfm_lowcmd_loop.cpp").read_text()

    construction_guard = _guard_containing(
        source, "std::make_shared<unitree::robot::ChannelPublisher<LowCmd>>")
    send_guard = _guard_containing(source, "publisher->Write(MakeLowCmd(command))")
    for guard in (construction_guard, send_guard):
        assert "args.dds_domain == kSafeDomain" in guard
        assert "IsLoopbackInterface(args.dds_interface)" in guard

    # There is exactly one LowCmd send and it is the send protected above.
    assert source.count("->Write(MakeLowCmd(command))") == 1
