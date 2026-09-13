"""Record root's completed source review and synthetic verification only."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

BASE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


prep_path = BASE / "source_preparation_audit_v2.json"
prep = json.loads(prep_path.read_text(encoding="utf-8-sig"))
assert sha(prep_path) == "7f22377e0230a8f7bd27f2ca0302381f238c7a2c5925a767eb36588ea2a98e26"
source = Path(prep["source_directory"])
actual = {p.name: sha(p) for p in sorted(source.glob("*.py"))}
assert actual == prep["source_sha256"] and len(actual) == 12
original_decoder = BASE.parent / "independent_native_stepper_saved_audit_v1/source_v2/saved_math.py"
assert sha(original_decoder) == actual["qualified_binary_math.py"]
tests = BASE / "root_audit_synthetic_tests_v2.xml"
suites = ET.parse(tests).getroot().findall("testsuite")
assert sum(int(s.attrib["tests"]) for s in suites) == 43
assert all(int(s.attrib[k]) == 0 for s in suites for k in ("failures", "errors", "skipped"))
reference = BASE.parent / "independent_native_stepper_v1/source_draft_v3/bfm_observations.py"
result = {
    "kind": "root_independent_saved_clock_source_review",
    "version": 2,
    "passed": True,
    "source_sha256": actual,
    "preparation_subject": {"path": str(prep_path), "sha256": sha(prep_path)},
    "root_synthetic_tests": {"passed": 43, "failed": 0, "skipped": 0,
                             "path": str(tests), "sha256": sha(tests)},
    "synthetic_observation_reference": {"path": str(reference), "sha256": sha(reference)},
    "reviewed": [
        "All 12 final source files; unchanged binary decoder verified against previously reviewed source.",
        "Literal source, request, launch, clearance, owner, process and output subject binding with final rehash.",
        "Independent full291/packed373 reconstruction, unchanged manual PD and original strict predicates.",
        "Nonfinite captured state/force retained as strict failure evidence; original initial force exception preserved.",
        "Fixed epoch and deadlines, actual plant-copy admission, worker timing, canonical jobs and command bytes.",
        "Incoming applied prior, lagged history and held-command provenance; no oracle replacement of actual state.",
        "Native attempt/return/capture/verification counts and reserved uncommitted fault bytes without full-scope credit.",
        "Four actual saved MJB buffers compared directly without additional serialization.",
        "Worker cleanup and observed process absence; failed-start uncertainty retained.",
        "Physical, timing, command and evidence-integrity verdicts remain separate."
    ],
    "limitations": [
        "Source/synthetic review only; actual benchmark and saved audit have not run.",
        "An uncommitted failed cycle can lack timing samples; incomplete evidence cannot qualify the component.",
        "Recorded expert command benchmark cannot qualify learned online policy or live Pico teleoperation."
    ],
    "actual_task_array_evaluations": 0,
    "native_steps": 0,
    "model_calls": 0,
    "optimizer_updates": 0
}
destination = BASE / "root_source_review_v2.json"
with destination.open("x", encoding="utf-8") as stream:
    json.dump(result, stream, indent=2, allow_nan=False)
    stream.write("\n")
print(json.dumps({"path": str(destination), "sha256": sha(destination), "passed": True}))
