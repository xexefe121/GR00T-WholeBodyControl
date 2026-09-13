"""Serial offline checks. No active robot executable or network client runs."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path.cwd().resolve()
HERE = Path(__file__).resolve().parent
OUT = HERE / "validated"
TEST = "gear_sonic/tests/test_g1_true23_restore_rpc_no_robot.py"
MODULES = (
    "gear_sonic/scripts/audit_g1_true23_restore_rpc_no_robot.py",
    "gear_sonic/scripts/qualify_g1_true23_active_lifecycle_no_robot.py",
    TEST,
)
CPP = "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


paths = [ROOT / path for path in MODULES]
paths += [ROOT / CPP / path for path in (
    "tests/true23_restore_rpc_driver.cpp", "tests/true23_restore_rpc_fixture.hpp",
    "src/g1_true23_active_gantry.cpp", "include/true23_active_gantry_core.hpp",
    "include/true23_live_shadow_core.hpp", "include/true23_shadow_gate.hpp",
)]
paths += [Path(__file__).resolve(), HERE / "qualifier_before.py"]
hashes = {str(path): sha(path) for path in paths}
stages = []


def run(name, command, expected, extra_env=None):
    start = time.monotonic()
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    env.update(extra_env or {})
    print(f"starting {name}", flush=True)
    with (OUT / f"{name}.log").open("x") as stream:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, check=False, timeout=90)
    stages.append(dict(name=name, command=command, returncode=result.returncode,
                       expected_returncode=expected, elapsed_s=time.monotonic() - start,
                       log_sha256=sha(OUT / f"{name}.log")))
    print(json.dumps(stages[-1]), flush=True)
    assert result.returncode == expected, name


def main():
    OUT.mkdir(exist_ok=False)
    run("critical_lint", [sys.executable, "-m", "ruff", "check", "--select", "E,F", *MODULES], 0)
    run("format_check", [sys.executable, "-m", "ruff", "format", "--check", *MODULES], 0)
    run("current_contract_tests", [sys.executable, "-m", "pytest", "-q", "-x", "--tb=short",
        "--basetemp=" + str(OUT / "current_tests_contract_v2"), TEST], 0)
    run("committed_contract_tests", [sys.executable, "-m", "pytest", "-q", "-x", "--tb=short",
        "--basetemp=" + str(OUT / "committed_tests_contract_v2"), TEST], 0,
        {"G1_TRUE23_RESTORE_TEST_RUNTIME_ROOT": str(HERE / "committed_source")})
    run("qualifier_before", [sys.executable, str(HERE / "qualifier_before.py"),
        "--repository-root", str(ROOT), "--output", str(OUT / "qualification_before.json")], 0)
    run("qualifier_after", [sys.executable, "-m", "gear_sonic.scripts.qualify_g1_true23_active_lifecycle_no_robot",
        "--repository-root", str(ROOT), "--output", str(OUT / "qualification_after.json")], 2)
    before = json.loads((OUT / "qualification_before.json").read_text())
    after = json.loads((OUT / "qualification_after.json").read_text())
    assert before["passed"] is True and after["passed"] is False
    assert after["dds_opened"] is False and after["robot_commands_published"] is False
    assert after["active_controller_launched"] is False
    assert after["checks"]["core_harness_passed"] is True
    assert after["checks"]["surface_audit_passed"] is True
    for path, digest in hashes.items():
        assert sha(path) == digest, path
    report = dict(experiment_completed=True, stages=stages, inputs=hashes,
                  old_qualification_reported_pass=before["passed"],
                  updated_qualification_reported_pass=after["passed"],
                  old_scope="Core/binary surface only; did not execute restore RPC routine",
                  new_rpc_contract_checks=after["checks"]["restore_rpc_contract"],
                  hardware_authorized=False, deployment_ready=False,
                  robot_commands_published=False, dds_opened=False)
    with (OUT / "verification.json").open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
