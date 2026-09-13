"""Adapt complete planned29 choreography from a saved reference/rollout trace.

Only planned_qpos50 is consumed. Recorded policy poses (pre_qpos/post_qpos)
are never substituted for the requested choreography. No robot interfaces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.utils import g1_23dof_task_space_retarget as ik
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_retarget import (
    AdaptationLimits,
    adapt_offline_motion,
    refine_retained_protected_motion,
)
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256


def planned_named_source(trace, source_model):
    planned = np.asarray(trace["planned_qpos50"], dtype=np.float64)
    times = np.asarray(trace["command_time_s"], dtype=np.float64)
    if planned.ndim != 2 or planned.shape[1] != 36 or len(planned) < 2:
        raise ValueError("complete planned_qpos50 must be [N,36]")
    if times.shape != (len(planned),) or not np.allclose(np.diff(times), 0.02, atol=1e-9, rtol=0):
        raise ValueError("planned choreography must have exact complete 50-Hz timestamps")
    if not np.isfinite(times).all() or not np.isfinite(planned).all():
        raise ValueError("planned choreography contains nonfinite values")
    return {
        "joint_names": np.asarray(ik._model_layout(source_model).joint_names),
        "joint_pos": planned[:, 7:].copy(),
        "root_pos_w": planned[:, :3].copy(),
        "root_quat_wxyz": planned[:, 3:7].copy(),
        "fps": np.array([50.0]),
        "timestamps_s": times.copy(),
    }


def planned_adaptation_options(*, root_reference_refinement: bool, preserve_source_excursion: bool = False):
    """Root refinement is one bounded diagnostic, never a larger hidden sweep."""
    if type(root_reference_refinement) is not bool:
        raise ValueError("root_reference_refinement must be boolean")
    if type(preserve_source_excursion) is not bool:
        raise ValueError("preserve_source_excursion must be boolean")
    if preserve_source_excursion and not root_reference_refinement:
        raise ValueError("full source excursion requires the single bounded root-refinement candidate")
    return {
        "limits": (
            AdaptationLimits(duration_scales=(2.0,), excursion_scales=(1.0 if preserve_source_excursion else 0.9,))
            if root_reference_refinement
            else AdaptationLimits()
        ),
        "root_reference_refinement": root_reference_refinement,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--target-model", type=Path, default=Path(ik.DEFAULT_TARGET_MODEL))
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--root-reference-refinement",
        action="store_true",
        help="Run one constrained root+23 reference refinement at 2x duration/90%% excursion",
    )
    parser.add_argument(
        "--protected-root-refinement-from",
        type=Path,
        help="One hard-constrained refinement of a hash-bound rejected diagnostic report",
    )
    parser.add_argument(
        "--preserve-source-excursion",
        action="store_true",
        help="With --root-reference-refinement, fit one 2x-duration/full-excursion source; no amplitude reduction",
    )
    parser.add_argument(
        "--feasibility-restoration",
        action="store_true",
        help="With a retained rejected diagnostic, recover intermediate feasibility; final bounds stay unchanged",
    )
    args = parser.parse_args(argv)
    if args.root_reference_refinement and args.protected_root_refinement_from is not None:
        parser.error("choose weighted root refinement or retained hard refinement, not both")
    if args.preserve_source_excursion and not args.root_reference_refinement:
        parser.error("--preserve-source-excursion requires --root-reference-refinement")
    if args.feasibility_restoration and args.protected_root_refinement_from is None:
        parser.error("--feasibility-restoration requires --protected-root-refinement-from")
    output = args.output_directory.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    paths = [
        args.trace,
        args.source_model,
        args.target_model,
        Path(__file__),
        Path(ik.__file__),
        Path(__file__).resolve().parents[1] / "utils/g1_true23_generalist_retarget.py",
    ]
    helper_root = Path(__file__).resolve().parents[1] / "utils"
    paths.extend(
        helper_root / filename
        for filename in (
            "g1_true23_original_task_trajectory.py",
            "g1_true23_box_qp.py",
            "g1_23dof_trajectory_projection.py",
            "g1_23dof_safe_target_transform.py",
            "g1_23dof_contract.py",
            "g1_true23_reference_floor.py",
            "g1_true23_generalist_corpus.py",
            "g1_true23_generalist_protected_root.py",
            "g1_true23_generalist_feasibility_restore.py",
        )
    )
    bindings = {str(path.resolve(strict=True)): sha256_file(path) for path in paths}
    forensic_report, stored = None, None
    if args.protected_root_refinement_from is not None:
        forensic_path = args.protected_root_refinement_from.resolve(strict=True)
        forensic_hash = sha256_file(forensic_path)
        forensic_report = json.loads(forensic_path.read_text())
        artifact = forensic_report["diagnostic_artifact"]
        if (
            artifact.get("accepted_motion") is not False
            or artifact.get("motion_schema_compatible") is not False
            or Path(artifact["path"]).name != artifact["path"]
        ):
            raise ValueError("retained input must be a rejected diagnostic-only local artifact")
        diagnostic_path = (forensic_path.parent / artifact["path"]).resolve(strict=True)
        if diagnostic_path.parent != forensic_path.parent or sha256_file(diagnostic_path) != artifact["sha256"]:
            raise ValueError("retained diagnostic path/hash differs from its forensic receipt")
        for path in (
            args.trace,
            args.source_model,
            args.target_model,
            Path(ik.__file__),
            helper_root / "g1_true23_original_task_trajectory.py",
            helper_root / "g1_23dof_safe_target_transform.py",
            helper_root / "g1_23dof_contract.py",
        ):
            key = str(path.resolve(strict=True))
            if forensic_report["input_bindings"].get(key) != bindings[key]:
                raise ValueError(f"retained source/model/baseline helper binding differs: {key}")
        with np.load(diagnostic_path, allow_pickle=False) as archive:
            stored = {key: archive[key].copy() for key in archive.files}
        bindings[str(forensic_path)] = forensic_hash
        bindings[str(diagnostic_path)] = artifact["sha256"]
    source = (
        mujoco.MjModel.from_binary_path(str(args.source_model))
        if args.source_model.suffix == ".mjb"
        else mujoco.MjModel.from_xml_path(str(args.source_model))
    )
    target = mujoco.MjModel.from_xml_path(str(args.target_model))
    if forensic_report is not None and forensic_report["compiled_models"] != {
        "source": compiled_model_sha256(source),
        "target": compiled_model_sha256(target),
    }:
        raise ValueError("retained compiled model identity differs")
    with np.load(args.trace, allow_pickle=False) as archive:
        arrays = planned_named_source(archive, source)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "planned.named29.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    print(f"Adapting all {len(arrays['joint_pos'])} planned source frames", flush=True)
    try:
        if forensic_report is not None:
            result = refine_retained_protected_motion(
                source_model=source,
                target_model=target,
                arrays=arrays,
                stored=stored,
                forensic_report=forensic_report,
                feasibility_restoration=args.feasibility_restoration,
            )
        else:
            result = adapt_offline_motion(
                source_model=source,
                target_model=target,
                arrays=arrays,
                source_role="requested_choreography",
                **planned_adaptation_options(
                    root_reference_refinement=args.root_reference_refinement,
                    preserve_source_excursion=args.preserve_source_excursion,
                ),
            )
        report = result.report
        if result.diagnostic_arrays is not None:
            diagnostic_path = output / "solver.diagnostic-only.npz"
            with diagnostic_path.open("xb") as stream:
                np.savez_compressed(stream, **result.diagnostic_arrays)
            report["diagnostic_artifact"] = {
                "path": diagnostic_path.name,
                "sha256": sha256_file(diagnostic_path),
                "accepted_motion": False,
                "motion_schema_compatible": False,
                "use": "read_only_solver_forensics_not_training_or_deployment",
            }
        if result.accepted:
            with (output / "adapted.true23.npz").open("xb") as stream:
                np.savez_compressed(stream, **result.arrays)
            report["adapted_motion_sha256"] = sha256_file(output / "adapted.true23.npz")
    except ValueError as exc:
        report = {"accepted": False, "input_error": str(exc), "dynamic_feasibility_verified": False}
    report.update(
        input_bindings=bindings,
        source_field="planned_qpos50",
        source_frames=len(arrays["joint_pos"]),
        recorded_policy_pose_used_as_choreography=False,
        root_reference_refinement_requested=args.root_reference_refinement or forensic_report is not None,
        hard_protected_refinement_requested=forensic_report is not None,
        intermediate_feasibility_restoration_requested=args.feasibility_restoration,
        named_source_sha256=sha256_file(output / "planned.named29.npz"),
        compiled_models={"source": compiled_model_sha256(source), "target": compiled_model_sha256(target)},
        hardware_authorized=False,
        deployment_ready=False,
    )
    for path, expected in bindings.items():
        if sha256_file(Path(path)) != expected:
            raise ValueError(f"input changed during retarget: {path}")
    with (output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
    print(output / "report.json", flush=True)
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
