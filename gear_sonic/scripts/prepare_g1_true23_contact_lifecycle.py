"""Bind a contact-conditioned diagnostic to a fresh calibrated SIM lifecycle.

Contact conditioning can shift the first pelvis location. Final offline start
calibration is one fixed transform of this NEW reference; no runtime reanchor,
old benchmark equivalence, or accepted training membership is claimed.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from gear_sonic.scripts import register_g1_true23_motion_start as registration
from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contact-report", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_directory.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError("contact lifecycle preparation refuses overwrite")
    report_path = args.contact_report.resolve(strict=True)
    report = json.loads(report_path.read_text())
    if (
        report.get("kind") != "g1_true23_stance_foot_cleanup_experiment_v1"
        or report.get("provisional_geometry_screen_passed") is not True
        or report.get("training_reference_accepted") is not False
        or report.get("hardware_authorized") is not False
        or report.get("deployment_ready") is not False
    ):
        raise ValueError("requires explicitly unqualified contact experiment with passing geometry")
    source_path = (report_path.parent / report["output"]["path"]).resolve(strict=True)
    if source_path.parent != report_path.parent or sha256_file(source_path) != report["output"]["sha256"]:
        raise ValueError("contact-conditioned output binding changed")
    originals = [
        (Path(p), h) for p, h in report["inputs"].items() if Path(p).name == "registered.source.diagnostic.npz"
    ]
    if len(originals) != 1 or sha256_file(originals[0][0]) != originals[0][1]:
        raise ValueError("contact experiment must bind its original named native23 source")
    with np.load(originals[0][0], allow_pickle=False) as archive:
        if tuple(archive["joint_names"].tolist()) != tuple(HARDWARE_23_JOINT_NAMES):
            raise ValueError("contact source physical joint identity differs")
    with np.load(source_path, allow_pickle=False) as archive:
        expected = {
            "fps",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        }
        if set(archive.files) != expected:
            raise ValueError("contact output must retain complete physical motion channels")
        named = {key: archive[key].copy() for key in archive.files}
    named["joint_names"] = np.asarray(HARDWARE_23_JOINT_NAMES)
    output.mkdir(parents=True, exist_ok=False)
    named_path = output / "contact_conditioned.named23.npz"
    with named_path.open("xb") as stream:
        np.savez_compressed(stream, **named)
    receipt = dict(
        kind="g1_true23_contact_reference_named_wrapper_v1",
        inputs={
            str(report_path): sha256_file(report_path),
            str(source_path): sha256_file(source_path),
            str(originals[0][0]): originals[0][1],
            str(Path(__file__).resolve()): sha256_file(Path(__file__)),
        },
        named_source_sha256=sha256_file(named_path),
        physical_motion_arrays_changed=False,
        final_fixed_source_calibration_after_contact_conditioning=True,
        historical_benchmark_or_single_registration_transition_claimed=False,
        remaining_conditional_force_support_failures=report["reference_support"][
            "frames_with_no_support_solution"
        ],
        training_reference_accepted=False,
        hardware_authorized=False,
        deployment_ready=False,
    )
    with (output / "source_receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    return registration.main(
        [
            "--source-motion",
            str(named_path),
            "--asset-root",
            str(args.asset_root),
            "--output-directory",
            str(output / "registered"),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
