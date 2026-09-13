"""Freeze one durable terminal-hold command; no execution."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
PROCESS = NEW / "pico_terminal_hold_v1_process"
OUTPUT = NEW / "pico_terminal_hold_v1"
if OUTPUT.exists() or (PROCESS / "pid.json").exists():
    raise SystemExit("run already exists; never duplicate")
PROCESS.mkdir(exist_ok=True)
launcher = (NEW / "launch_pico_full_control_lm_v1.ps1").read_text()
launcher = launcher.replace("pico_control_lm_integration_v1/repo", "pico_terminal_hold_source_v1/repo")
launcher = launcher.replace("pico_full_control_lm_v1", "pico_terminal_hold_v1")
launcher = launcher.replace("'--checkpoint-controls', '100',", "'--checkpoint-controls', '50',")
launcher = launcher.replace("    '--output',", "    '--preceding-run', '/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/pico_full_control_lm_v1',\n    '--preceding-endpoint', '/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/pico_full_endpoint_independent_v1/endpoint.npz',\n    '--output',")
launcher_path = NEW / "launch_pico_terminal_hold_v1.ps1"
launcher_path.write_text(launcher)
wrapper = (NEW / "pico_full_control_lm_v1_process/run_durable.ps1").read_text()
wrapper = wrapper.replace("pico_full_control_lm_v1", "pico_terminal_hold_v1")
wrapper = wrapper.replace("requested_controls=6530", "requested_controls=250")
wrapper = wrapper.replace("$report.completed_controls -eq 6530", "$report.completed_controls -eq 250 -and $report.boundary_verified")
wrapper_path = PROCESS / "run_durable.ps1"
wrapper_path.write_text(wrapper)
prior = json.loads((NEW / "pico_full_control_lm_v1_process/launch_receipt.json").read_text())
paths = [Path(path) for path in prior["verified_hashes"]]
paths += [path for path in (HERE / "repo").rglob("*.py")]
paths += [HERE / name for name in ("continuity.py", "build_adapter.py", "adapter.diff", "source_transformations.json",
                                    "frozen_source_hashes.json", "test_continuity.py", "run_tests.py", "focused_tests_v3.xml",
                                    "preflight.py", "preflight.json", "prepare_launcher.py")]
paths += [NEW / "pico_full_control_lm_v1" / name for name in ("trace.npz", "trace.partial.npz", "report.json", "request.json")]
paths += [NEW / "pico_full_endpoint_independent_v1" / name for name in ("endpoint.npz", "report.json")]
paths += [launcher_path, wrapper_path]
def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
hashes = {str(path): sha(path) for path in paths}
for path, expected in prior["verified_hashes"].items():
    if hashes[path] != expected:
        raise ValueError("previous source/input changed: " + path)
tests = ET.parse(HERE / "focused_tests_v3.xml").getroot()
suites = list(tests.iter("testsuite"))
counts = {key: sum(int(suite.attrib.get(key, 0)) for suite in suites) for key in ("tests", "failures", "errors", "skipped")}
if counts != dict(tests=25, failures=0, errors=0, skipped=0):
    raise ValueError("focused boundary checks failed: " + str(counts))
preflight = json.loads((HERE / "preflight.json").read_text())
if not preflight["passed"] or preflight["source_sha256"] != sha(HERE / "repo/gear_sonic/utils/pico_terminal_continuity.py"):
    raise ValueError("preflight stale or failed")
receipt = dict(utc=datetime.now(timezone.utc).isoformat(),
               scope="ONE same-MPC separate250 hold, preceded by exact6500-to6530 tail reexecution",
               checkpoint_control=6500, boundary_control=6530, requested_hold_controls=250,
               extension_global_stop=6780, exact_launcher_text=launcher,
               original_source_preserved=True, root_endpoint_check_only=True,
               boundary_gate="all30control fields/300steps bitexact and independently reconstructed291endpoint before any hold control",
               focused_tests=counts, saved_history_preflight=preflight, verified_hashes=hashes,
               output=str(OUTPUT), launcher=str(launcher_path), wrapper=str(wrapper_path),
               frozen_pythonpath=str(HERE / "repo"), hidden_wrapper_required=True,
               hardware_authorized=False)
(PROCESS / "launch_receipt.json").write_text(json.dumps(receipt, indent=2))
print(json.dumps(dict(launch_receipt_sha256=sha(PROCESS / "launch_receipt.json"), hashes=len(hashes),
                      launcher_sha256=sha(launcher_path), wrapper_sha256=sha(wrapper_path))))
