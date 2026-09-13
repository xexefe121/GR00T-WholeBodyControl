"""Preserve the strict failed input audit and inspect its first discrepancy."""

import json
from pathlib import Path
import runpy

import numpy as np

from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file

HERE = Path(__file__).resolve().parent


def main():
    output = HERE / "train500_v1/sampled_reference_failure.json"
    if output.exists():
        raise FileExistsError("reference failure diagnosis refuses overwrite")
    audit = HERE / "audit_sampled_references.py"
    try:
        runpy.run_path(str(audit), run_name="__main__")
    except AssertionError as error:
        frame = error.__traceback__
        while frame.tb_next is not None:
            frame = frame.tb_next
        values = frame.tb_frame.f_locals
        assert "orientation_error" in values and "velocity_error" not in values
        actual = values["tokenizer"][..., 262:]
        expected = values["orientation"]
        difference = np.max(np.abs(actual - expected), axis=-1)
        index = np.unravel_index(np.argmax(difference), difference.shape)
        reference = values["native"]["body_quat_w"][values["anchors"][index], 0]
        measured = values["measured_quat"][index]
        report = dict(
            kind="failed_strict_sampled_reference_orientation_diagnosis_v1",
            failed_audit_sha256=sha256_file(audit),
            diagnosis_script_sha256=sha256_file(Path(__file__)),
            original_orientation_tolerance=2e-6,
            maximum_error=values["orientation_error"],
            bad_rows=int((difference > 2e-6).sum()),
            total_rows=int(difference.size),
            error_p50_p95_p99=np.percentile(difference, [50, 95, 99]).tolist(),
            worst_sample_control=int(values["controls"][index[0]]),
            worst_env=int(index[1]),
            worst_reference_q0=int(values["anchors"][index]),
            actual_orientation6=actual[index].tolist(),
            expected_orientation6=expected[index].tolist(),
            measured_quaternion=measured.tolist(),
            source_quaternion=reference.tolist(),
            measured_quaternion_norm=float(np.linalg.norm(measured)),
            source_quaternion_norm=float(np.linalg.norm(reference)),
            prior_lower240_original_vr21_checks_passed=True,
            later_input_checks_not_reached=True,
            threshold_relaxed=False,
            dynamics_or_tracking_success_claimed=False,
            hardware_authorized=False,
            deployment_ready=False,
            inputs=values["pins"],
        )
        with output.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        print(json.dumps({k: v for k, v in report.items() if k != "inputs"}), flush=True)
    else:
        raise ValueError("strict audit unexpectedly passed; preserve and investigate")


if __name__ == "__main__":
    main()
