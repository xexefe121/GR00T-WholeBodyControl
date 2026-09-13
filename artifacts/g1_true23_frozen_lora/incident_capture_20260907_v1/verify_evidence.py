"""Bind completed local tests/benchmarks to current diagnostic source; no SDK."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from gear_sonic.utils.g1_true23_incident_capture import sha256

root = Path.cwd().resolve()
output = Path(__file__).resolve().parent
benchmark_path = output / "raw_first30/benchmark.json"
benchmark = json.loads(benchmark_path.read_text())
suite = ET.parse(output / "raw_first_tests_v1.xml").getroot().find("testsuite")
assert suite.attrib["tests"] == "73"
assert all(suite.attrib[key] == "0" for key in ("errors", "failures", "skipped"))
assert benchmark["capture_integrity_pass"] is True
assert benchmark["generated_pairs"] == 15000
assert benchmark["capture"]["collector_dropped_packets"] == {}
assert benchmark["dds_transport_tested"] is False
for path, digest in {**benchmark["source_sha256"], **benchmark["interpretation"]["decoder_source_sha256"]}.items():
    assert sha256(Path(path)) == digest, path
sources = [
    "gear_sonic/utils/g1_true23_incident_capture.py",
    "gear_sonic/scripts/record_g1_true23_incident_readonly.py",
    "gear_sonic/scripts/decode_g1_true23_incident_no_robot.py",
    "gear_sonic/scripts/benchmark_g1_true23_incident_capture_no_robot.py",
    "gear_sonic/tests/test_g1_true23_incident_capture.py",
    "gear_sonic/tests/test_g1_true23_incident_readonly_sdk.py",
    "gear_sonic/tests/test_g1_true23_incident_decode.py",
]
protected = {
    "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_true23_active_gantry.cpp": "7c94f882ad4e7faccd8ba1de6571167fbf82f2c9fc12eaa6f9b7f0c00a017ddf",
    "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/true23_active_gantry_core.hpp": "5c7965251b49d4f3005e9802aebddb03f60534603992c11837078e5ac41badaf",
}
assert all(sha256(root / path) == digest for path, digest in protected.items())
evidence = ["raw_first_tests_v1.xml", "benchmark10/summary.json", "raw_first10/benchmark.json", "raw_first30/benchmark.json", "raw_first30/summary.json", "raw_first30/interpretation/decode_summary.json"]
report = {
    "kind": "g1_true23_incident_offline_verification_v1",
    "tests": {key: suite.attrib[key] for key in ("tests", "errors", "failures", "skipped", "time")},
    "source_sha256": {path: sha256(root / path) for path in sources},
    "evidence_sha256": {path: sha256(output / path) for path in evidence},
    "protected_hardware_source_unchanged": protected,
    "final_benchmark_callbacks": 30000,
    "final_benchmark_collector_drops": 0,
    "earlier_inline_decode_dropped_callbacks": 4365,
    "code_and_file_capture_verified": True,
    "dds_transport_tested": False,
    "physical_cause_proven": False,
    "hardware_authorized": False,
    "deployment_ready": False,
}
with (output / "verification.json").open("x", encoding="utf-8") as stream:
    json.dump(report, stream, indent=2)
print(json.dumps({"verification_sha256": sha256(output / "verification.json"), "tests": report["tests"], "deployment_ready": False}))
