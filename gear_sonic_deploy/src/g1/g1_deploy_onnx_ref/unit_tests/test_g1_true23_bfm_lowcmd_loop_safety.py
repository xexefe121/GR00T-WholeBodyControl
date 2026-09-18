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
