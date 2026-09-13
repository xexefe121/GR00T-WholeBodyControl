"""Negative evidence: malformed effort arrays and engine warnings cannot pass."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil

import numpy as np

from artifacts.teleop_six_hour_20260910.qualify_recorded_candidate import audit


def run():
    base = Path("E:/codex-artifacts/sonic23_teleop_six_hour_20260910/bfm_online_intent_v2")
    source = base / "pico_v4_prefix535_replay_v1"
    output = base / "recorded_auditor_negative_witness_v2"
    output.mkdir(exist_ok=False)
    report = json.loads((source / "report.json").read_text())
    with np.load(source / "trace.npz", allow_pickle=False) as archive:
        trace = {key: archive[key].copy() for key in archive.files}
    results = {}
    for name in ("valid_record", "broadcast_effort", "engine_warning", "missing_warnings"):
        case = output / name
        case.mkdir()
        altered_report = dict(report)
        if name == "engine_warning":
            altered_report["engine_warning_counts"] = [0, 0, 0, 0, 0, 1, 0, 0]
        elif name == "missing_warnings":
            altered_report.pop("engine_warning_counts", None)
        (case / "report.json").write_text(json.dumps(altered_report, indent=2))
        if name == "broadcast_effort":
            altered = dict(trace)
            altered["physics_torque"] = trace["physics_torque"][:, :1]
            np.savez_compressed(case / "trace.npz", **altered)
        else:
            shutil.copyfile(source / "trace.npz", case / "trace.npz")
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                result = audit(case, case / "audit.json")
            except ValueError as error:
                if name != "broadcast_effort" or str(error) != "physical trace dimensions differ":
                    raise
                results[name] = dict(rejected=True, reason=str(error))
                continue
        if name == "broadcast_effort":
            raise AssertionError("broadcastable malformed effort array was accepted")
        expected_warning_gate = name == "valid_record"
        assert result["gates"]["no_engine_warnings"] == expected_warning_gate
        assert not result["recorded_source_tracking_pass"]
        if name == "valid_record":
            baseline_failures = set(result["failed_gates"])
        expected_failures = baseline_failures.copy()
        if not expected_warning_gate:
            expected_failures.add("no_engine_warnings")
        assert set(result["failed_gates"]) == expected_failures
        results[name] = dict(failed_gates=result["failed_gates"],
                             trace_sha256=result["trace_sha256"],
                             auditor_sha256=result["audit_sha256"])
    (output / "report.json").write_text(json.dumps(dict(
        purpose="Deliberately altered audit fixtures, never controller evidence",
        source=str(source), source_trace_sha256=hashlib.sha256((source / "trace.npz").read_bytes()).hexdigest(),
        cases=results, passed=True), indent=2))
    shutil.copyfile(__file__, output / "witness_snapshot.py")
    shutil.copyfile(Path(__file__).with_name("qualify_recorded_candidate.py"), output / "auditor_snapshot.py")
    print(json.dumps(results))


if __name__ == "__main__":
    run()
