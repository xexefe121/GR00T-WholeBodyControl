"""Adapt one explicitly named 29-joint motion with bounded offline MuJoCo IK.

Input NPZ fields: joint_names[29], joint_pos[N,29], root_pos_w[N,3],
root_quat_wxyz[N,4], fps[1]; optional uniform timestamps_s[N] and boolean
contact_flags[N,2]. Output includes complete source-time correspondence and
separate original/adapted task errors. Rejected motions emit a report, no
accepted motion. This is not live teleoperation or dynamics qualification.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from gear_sonic.utils import g1_23dof_task_space_retarget as ik
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_retarget import AdaptationLimits, adapt_offline_motion
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, default=Path(ik.DEFAULT_SOURCE_MODEL))
    parser.add_argument("--target-model", type=Path, default=Path(ik.DEFAULT_TARGET_MODEL))
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--ik-iterations", type=int, default=16)
    parser.add_argument("--maximum-output-frames", type=int, default=30000)
    args = parser.parse_args(argv)
    output = args.output_directory.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")
    limits = AdaptationLimits(ik_iterations=args.ik_iterations, maximum_output_frames=args.maximum_output_frames)
    input_path = args.input.resolve(strict=True)
    bindings = {
        str(path.resolve(strict=True)): sha256_file(path.resolve(strict=True))
        for path in (
            input_path,
            args.source_model,
            args.target_model,
            Path(__file__),
            Path(__file__).resolve().parents[1] / "utils/g1_true23_generalist_retarget.py",
            Path(ik.__file__),
        )
    }
    source, target = ik.load_models(args.source_model, args.target_model)
    model_hashes = {"source": compiled_model_sha256(source), "target": compiled_model_sha256(target)}
    output.mkdir(parents=True, exist_ok=False)
    try:
        with np.load(input_path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        result = adapt_offline_motion(source_model=source, target_model=target, arrays=arrays, limits=limits)
        report = result.report
        for path, expected in bindings.items():
            if sha256_file(Path(path)) != expected:
                raise ValueError(f"retarget input or source changed during run: {path}")
        if result.accepted:
            motion_path = output / "adapted.true23.npz"
            with motion_path.open("xb") as stream:
                np.savez_compressed(stream, **result.arrays)
            report["output"] = {"path": motion_path.name, "sha256": sha256_file(motion_path)}
    except (ValueError, KeyError, OSError) as exc:
        report = {
            "schema_version": 1,
            "kind": "g1_true23_generalist_bounded_offline_retarget",
            "accepted": False,
            "input_error": f"{type(exc).__name__}: {exc}",
            "limits": asdict(limits),
            "causal_live_adapter": False,
            "dynamic_feasibility_verified": False,
            "hardware_authorized": False,
        }
    report.update(input_bindings=bindings, compiled_model_sha256=model_hashes)
    with (output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(output / "report.json")
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
