"""Index separately evaluated contact-conditioned SIM references; raw bank unchanged."""

import argparse
import json
from pathlib import Path

from gear_sonic.utils.g1_true23_contact_bank_reference import CONTACT_REPAIR_INDEX_KIND, load_contact_repair_member
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-report", type=Path, required=True)
    parser.add_argument("--contact-root", type=Path, required=True)
    parser.add_argument("--evaluation-directory-name", default="policy1200_evaluation")
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_directory.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError("contact bank preparation refuses overwrite")
    bank_path = args.bank_report.resolve(strict=True)
    bank = json.loads(bank_path.read_text())
    if bank.get("accepted_reference_bank") is not True:
        raise ValueError("contact experiment requires the immutable accepted original bank")
    repairs, evaluations = [], []
    inputs = {str(bank_path): sha256_file(bank_path)}
    for clip in bank["clips"]:
        directory = args.contact_root.resolve(strict=True) / clip["name"]
        contact_path = directory / "report.json"
        contact = json.loads(contact_path.read_text())
        registered_inputs = [
            (p, h) for p, h in contact["inputs"].items() if Path(p).name == "registered.source.diagnostic.npz"
        ]
        if len(registered_inputs) != 1:
            raise ValueError("contact source must bind exactly one original registered member")
        registered = directory / "lifecycle/registered"
        final_path = registered / "registered.source.diagnostic.npz"
        repair_path = registered / "repaired_ramps/report.json"
        evaluation_path = registered / args.evaluation_directory_name / "report.json"
        row = dict(
            name=clip["name"],
            original_source_motion_sha256=clip["source_sha256"],
            registered_input_path=registered_inputs[0][0],
            registered_input_sha256=registered_inputs[0][1],
            contact_report_path=str(contact_path),
            contact_report_sha256=sha256_file(contact_path),
            registered_source_path=str(final_path),
            source_motion_sha256=sha256_file(final_path),
            report_path=str(repair_path),
            report_sha256=sha256_file(repair_path),
        )
        _, bindings = load_contact_repair_member(row, clip)
        inputs.update(bindings)
        inputs.update({str(p): sha256_file(p) for p in (repair_path, evaluation_path)})
        repairs.append(row)
        evaluations.append(
            dict(name=clip["name"], report_path=str(evaluation_path), report_sha256=sha256_file(evaluation_path))
        )
    if any(sha256_file(Path(p)) != h for p, h in inputs.items()):
        raise ValueError("contact bank preparation inputs changed")
    output.mkdir(parents=True, exist_ok=False)
    flags = dict(
        raw_original_fidelity_acceptance_inherited=False,
        dynamic_feasibility_proven=False,
        simulator_qualified=False,
        hardware_authorized=False,
        deployment_ready=False,
    )
    products = {
        "lifecycle_repairs.json": dict(
            kind=CONTACT_REPAIR_INDEX_KIND, bank_report_sha256=sha256_file(bank_path), repairs=repairs, **flags
        ),
        "parent_evaluations.json": dict(
            kind="g1_true23_reference_bank_per_clip_evaluation_index_v1",
            bank_report_sha256=sha256_file(bank_path),
            evaluations=evaluations,
            **flags,
        ),
        "preparation.json": dict(
            kind="g1_true23_contact_conditioned_bank_preparation_v1",
            input_bindings=inputs,
            raw_bank_modified=False,
            **flags,
        ),
    }
    for name, report in products.items():
        with (output / name).open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(output), clips=len(repairs), raw_bank_modified=False, **flags)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
