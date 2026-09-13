"""Bind unchanged contact sources and separately evaluated generated stepping references."""

import argparse
import json
from pathlib import Path

from gear_sonic.scripts.evaluate_g1_true23_generalist_baselines import write_json
from gear_sonic.utils.g1_true23_contact_step_bank_reference import INDEX_KIND
from gear_sonic.utils.g1_true23_contact_step_lifecycle_reference import POLICY_KIND
from gear_sonic.utils.g1_true23_contact_step_transition import PROFILE
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-report", type=Path, required=True)
    parser.add_argument("--step-root", type=Path, required=True)
    parser.add_argument("--evaluation-directory-name", default="policy1200_evaluation")
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        help="Optional evaluated policy directory containing one subdirectory per bank member",
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_directory.resolve()
    if output.exists() or args.output_directory.is_symlink():
        raise FileExistsError("contact-step bank refuses overwrite")
    bank_path = args.bank_report.resolve(strict=True)
    bank = json.loads(bank_path.read_text())
    if bank.get("accepted_reference_bank") is not True:
        raise ValueError("contact-step bank requires an unchanged accepted original source bank")
    inputs, references, evaluations = {str(bank_path): sha256_file(bank_path)}, [], []
    for clip in bank["clips"]:
        directory = args.step_root.resolve(strict=True) / clip["name"]
        path, evaluated = directory / "report.json", directory / args.evaluation_directory_name / "report.json"
        if args.evaluation_root is not None:
            evaluated = args.evaluation_root.resolve(strict=True) / clip["name"] / "report.json"
        reference, policy = json.loads(path.read_text()), json.loads(evaluated.read_text())
        if (
            reference["timeline"]["generated_transition_profile"] != PROFILE
            or reference["geometry"]["provisional_geometry_screen_passed"] is not True
            or policy.get("kind") != POLICY_KIND
            or [row["case"] for row in policy["records"]] != ["nominal", "standing_push_x", "standing_push_y"]
        ):
            raise ValueError(
                "every stepping reference needs its own complete scheduled unmodified-policy comparison"
            )
        references.append(dict(name=clip["name"], report_path=str(path), report_sha256=sha256_file(path)))
        evaluations.append(
            dict(name=clip["name"], report_path=str(evaluated), report_sha256=sha256_file(evaluated))
        )
        inputs.update({str(p): sha256_file(p) for p in (path, evaluated)})
    output.mkdir(parents=True, exist_ok=False)
    flags = dict(
        raw_bank_modified=False,
        source_motion_or_speed_changed_by_step_branch=False,
        dynamic_feasibility_proven=False,
        hardware_authorized=False,
        deployment_ready=False,
    )
    write_json(
        output / "step_references.json",
        dict(
            kind=INDEX_KIND,
            profile=PROFILE,
            bank_report_sha256=sha256_file(bank_path),
            references=references,
            **flags,
        ),
    )
    write_json(
        output / "parent_evaluations.json",
        dict(
            kind="g1_true23_reference_bank_per_clip_evaluation_index_v1",
            bank_report_sha256=sha256_file(bank_path),
            evaluations=evaluations,
            **flags,
        ),
    )
    write_json(
        output / "preparation.json",
        dict(
            kind="g1_true23_contact_step_bank_preparation_v1",
            input_bindings=inputs,
            original_report_kinds_or_eligibility_labels_rewritten=False,
            explicit_changed_reference_training_adoption_required=True,
            **flags,
        ),
    )
    print(json.dumps(dict(output=str(output), clips=len(references), **flags)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
