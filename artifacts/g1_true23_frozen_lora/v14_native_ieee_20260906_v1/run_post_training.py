"""Keep memory-heavy tests, ONNX reconstruction and evaluations serial."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


stages = [
    ("new_tests", [sys.executable, "-m", "pytest", "-q",
                   "gear_sonic/tests/test_train_g1_true23_v14_native_ieee.py",
                   "gear_sonic/tests/test_g1_true23_v14_diagnostic_pair.py",
                   f"--junitxml={HERE / 'new_tests.xml'}"]),
    ("paired_full_evaluation", [sys.executable, str(HERE / "export_evaluate.py")]),
    ("regressions", [sys.executable, str(HERE / "run_regressions.py")]),
]
report = []
assert (HERE / "experiment_report.json").is_file()
for name, command in stages:
    print(json.dumps(dict(starting_stage=name)), flush=True)
    log = HERE / f"{name}.log"
    start = time.monotonic()
    with log.open("x") as stream:
        code = subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
    row = dict(name=name, command=command, return_code=code, elapsed_s=time.monotonic() - start,
               log=str(log), log_sha256=digest(log))
    with (HERE / f"{name}.stage.json").open("x") as stream:
        json.dump(row, stream, indent=2)
        stream.write("\n")
    report.append(row)
    print(json.dumps(dict(completed_stage=name, return_code=code)), flush=True)
    if code:
        raise RuntimeError(f"v14 post-training failed at {name}; inspect retained {log}")
with (HERE / "post_training_report.json").open("x") as stream:
    json.dump(dict(stages=report, hardware_authorized=False, deployment_ready=False), stream, indent=2)
    stream.write("\n")
