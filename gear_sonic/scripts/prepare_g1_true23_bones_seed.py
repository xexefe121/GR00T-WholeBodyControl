"""Index existing local BONES-SEED files or convert one complete named29 source.

No downloads, license acceptance, training, robot control or implicit overwrite.
The index freezes capture-group splits before any data adaptation or selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gear_sonic.utils.g1_true23_bones_seed import (
    build_source_index,
    load_named_source_csv,
    resolve_csv_path,
    select_training_sources,
    write_json,
)
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    index = commands.add_parser("index")
    index.add_argument("--metadata", type=Path, required=True)
    index.add_argument("--extracted-root", type=Path, required=True)
    index.add_argument("--split-seed", default="g1_true23_bones_seed_original_takes_v1")
    convert = commands.add_parser("convert-source")
    convert.add_argument("--index-directory", type=Path, required=True)
    convert.add_argument("--motion-id", required=True)
    convert.add_argument("--source-model", type=Path, required=True)
    select = commands.add_parser("select-training")
    select.add_argument("--index-directory", type=Path, required=True)
    select.add_argument("--family", action="append", required=True)
    select.add_argument("--per-family", type=int, default=3)
    select.add_argument("--min-source-frames", type=int, default=360)
    select.add_argument("--max-source-frames", type=int, default=1440)
    select.add_argument("--selection-seed", default="g1_true23_bones_seed_curriculum_probe_v1")
    for command in (index, convert, select):
        command.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_directory.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")
    if args.mode == "index":
        rows, splits, report = build_source_index(args.metadata, args.extracted_root, split_seed=args.split_seed)
        output.mkdir(parents=True, exist_ok=False)
        index_path = output / "source_index.jsonl"
        with index_path.open("x", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        write_json(output / "recording_splits.json", splits)
        report["index"] = {"path": index_path.name, "sha256": sha256_file(index_path)}
        write_json(output / "report.json", report)
    else:
        index_dir = args.index_directory.resolve(strict=True)
        index_report = json.loads((index_dir / "report.json").read_text())
        if index_report.get("kind") != "g1_true23_bones_seed_local_source_index_v1":
            raise ValueError("not a BONES-SEED source index")
        index_path = index_dir / "source_index.jsonl"
        if sha256_file(index_path) != index_report["index"]["sha256"]:
            raise ValueError("source index bytes changed")
        rows = []
        with index_path.open(encoding="utf-8") as stream:
            for line in stream:
                rows.append(json.loads(line))
        if args.mode == "select-training":
            report = select_training_sources(
                rows,
                families=args.family,
                per_family=args.per_family,
                min_frames=args.min_source_frames,
                max_frames=args.max_source_frames,
                seed=args.selection_seed,
            )
            report["index_report_sha256"] = sha256_file(index_dir / "report.json")
            report["split_sha256"] = index_report["split_sha256"]
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / "report.json", report)
            print(output / "report.json")
            return 0
        import numpy as np
        from gear_sonic.utils import g1_23dof_task_space_retarget as ik

        matches = [row for row in rows if row["motion_id"] == args.motion_id]
        if len(matches) != 1:
            raise ValueError("motion ID must match exactly one indexed source")
        row = matches[0]
        if row["disk_status"] != "nonempty":
            raise ValueError("selected CSV was missing or empty during source indexing")
        selected_path, _ = resolve_csv_path(
            Path(index_report["extracted_root"]),
            {
                "move_g1_path": row["source_relative_path"],
                "take_date": row["capture"]["take_date"],
                "filename": Path(row["source_relative_path"]).stem,
            },
        )
        if str(selected_path) != row["source_path"] or selected_path.stat().st_size != row["source_size_bytes"]:
            raise ValueError("selected source path or size changed after indexing")
        model = ik.mujoco.MjModel.from_xml_path(str(args.source_model.resolve(strict=True)))
        arrays, receipt = load_named_source_csv(
            Path(row["source_path"]),
            expected_frames=row["source_frames"],
            joint_names=ik._model_layout(model).joint_names,
        )
        output.mkdir(parents=True, exist_ok=False)
        destination = output / "source.named29.npz"
        with destination.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        report = {
            "kind": "g1_true23_bones_seed_named29_source_v1",
            "indexed_source": row,
            "source": receipt,
            "index_report_sha256": sha256_file(index_dir / "report.json"),
            "split_sha256": index_report["split_sha256"],
            "source_model": {"path": str(args.source_model.resolve()), "sha256": sha256_file(args.source_model)},
            "output": {"path": destination.name, "sha256": sha256_file(destination)},
            "native23_retarget_applied": False,
            "training_corpus_ready": False,
            "archive_membership_verified": False,
            "hardware_authorized": False,
            "deployment_ready": False,
        }
        write_json(output / "report.json", report)
    print(output / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
