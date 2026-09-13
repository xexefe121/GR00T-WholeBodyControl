"""The Windows worktree must preserve the executable Bash receiver launcher."""

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "install_scripts/run_pico_robotics_service.sh"


def test_pico_service_launcher_uses_lf_and_preserves_checkout_rule():
    assert b"\r" not in LAUNCHER.read_bytes()
    attributes = (ROOT / ".gitattributes").read_text().splitlines()
    assert "install_scripts/run_pico_robotics_service.sh text eol=lf" in attributes


def test_pico_service_launcher_bash_syntax_without_starting_receiver():
    result = subprocess.run(["bash", "-n", str(LAUNCHER)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
