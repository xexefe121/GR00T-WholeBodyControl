"""All original SONIC clips, bounded pelvis-attitude/native23 diagnostic fit."""

from dataclasses import asdict
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from gear_sonic.scripts import build_g1_true23_sonic_library_motions as library
from gear_sonic.scripts.refine_g1_true23_reference_forces import bind_identity, dump
from gear_sonic.scripts.refine_g1_true23_stance_contacts import load_motion
from gear_sonic.scripts.retarget_g1_true23_stance import audit_rebuilt_causal_terms
from gear_sonic.utils import g1_23dof_task_space_retarget as retarget, g1_true23_original_task_trajectory as fit
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256
from gear_sonic.utils.g1_true23_sim_acquisition import audit_reference_kinematics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREVIOUS = HERE.parent / "original_planner_projected_seed_20260906_v1"
PLANNER = ROOT.parent / "GR00T-WholeBodyControl/artifacts/g1_true23/sonic_library_motion_suite_v2"
NAMES = {
    "original_sonic_hand_crawl": "hand_crawling",
    "original_sonic_elbow_crawl": "elbow_crawling",
    "original_sonic_happy_dance": "happy_dance",
}
FLAGS = dict(teacher_accepted=False, hardware_authorized=False, deployment_ready=False)


def main():
    if (HERE / "started.json").exists():
        raise FileExistsError("use a new immutable output directory")
    previous = json.loads((PREVIOUS / "report.json").read_text())
    identities = dict(previous["inputs"])
    if previous["clips_written"] != 3 or not previous["all_attempts_completed"]:
        raise ValueError("complete original-source projected seed is required")
    for path in list(identities):
        bind_identity(identities, path)
    for path in (
        PREVIOUS / "report.json",
        PREVIOUS / "motions.json",
        Path(__file__),
        Path(inspect.getfile(fit)),
        ROOT / "gear_sonic/utils/g1_true23_box_qp.py",
        ROOT / "gear_sonic/utils/g1_true23_sim_acquisition.py",
        ROOT / "gear_sonic/scripts/retarget_g1_true23_stance.py",
    ):
        bind_identity(identities, path)
    source, target = retarget.load_models(
        ROOT / retarget.DEFAULT_SOURCE_MODEL, ROOT / retarget.DEFAULT_TARGET_MODEL
    )
    hashes = {"source": compiled_model_sha256(source), "target": compiled_model_sha256(target)}
    if hashes != previous["compiled_models"]:
        raise ValueError("compiled models changed")
    config = fit.OriginalTaskConfig(maximum_iterations=24)
    dump(
        HERE / "started.json",
        {"inputs": identities, "compiled_models": hashes, "config": asdict(config), "complete": False, **FLAGS},
    )
    records = []
    for name, raw_name in NAMES.items():
        original = load_motion(PLANNER / f"{raw_name}.npz")
        xyz, quat, joints = library._resample_qpos(original["qpos"])
        seed = load_motion(PREVIOUS / f"{name}.npz")
        record = {"name": name, "frames_requested": len(xyz), "controlled_joints": 23, **FLAGS}
        print(json.dumps({"starting_original_clip": name, "frames": len(xyz)}), flush=True)
        try:
            problem = fit.OriginalTaskPath(
                source, target, np.column_stack((xyz, quat, joints)), seed["joint_pos"], config=config
            )

            def progress(row):
                print(
                    json.dumps(
                        {
                            "clip": name,
                            "iteration": row["iteration"],
                            "accepted": row["accepted"],
                            "merit": row.get("merit"),
                            "fraction": row.get("fraction"),
                            "qp_status": row["qp"]["status"],
                            "qp_row_violation": row["qp"].get("independent_maximum_scaled_linear_violation"),
                        }
                    ),
                    flush=True,
                )

            variables, evidence = fit.fit_original_task_path(problem, progress=progress)
            output = HERE / f"{name}.npz"
            with output.open("xb") as stream:
                np.savez_compressed(stream, **problem.serialize(variables))
            arrays = load_motion(output)
            saved = problem.serialized_variables(arrays)
            audit = problem.audit(saved)
            fk = audit_reference_kinematics(SimpleNamespace(module=mujoco, model=target), arrays)
            if not audit["passed"] or not fk["position_fk_consistent"] or not fk["orientation_fk_consistent"]:
                raise ValueError("serialized SE3 fit failed path/FK audit; not a usable reference")
            bind_identity(identities, output)
            record.update(
                output=str(output),
                output_sha256=identities[str(output)],
                fit=evidence,
                saved_path_audit=audit,
                saved_task_metrics=problem.metrics(saved),
                fk=fk,
                causal=audit_rebuilt_causal_terms(arrays),
                failure=None,
            )
        except (ValueError, RuntimeError) as error:
            record.update(output=None, failure={"type": type(error).__name__, "message": str(error)})
        records.append(record)
        dump(HERE / f"{name}.report.json", record)
        print(
            json.dumps(
                {
                    "clip": name,
                    "output": record["output"],
                    "failure": record["failure"],
                    "saved_metrics": record.get("saved_task_metrics"),
                }
            ),
            flush=True,
        )
    if all(row["output"] for row in records):
        previous_manifest = json.loads((PREVIOUS / "motions.json").read_text())
        outputs = {row["name"]: row for row in records}
        entries = [
            {**entry, "path": outputs[entry["name"]]["output"], "sha256": outputs[entry["name"]]["output_sha256"]}
            if entry["name"] in outputs
            else entry
            for entry in previous_manifest["motions"]
        ]
        dump(
            HERE / "motions.json",
            {
                "kind": "g1_true23_original_planner_se3_task_fit_diagnostic_manifest_v1",
                "motions": entries,
                "unchanged_pico_clip_count": 5,
                "diagnostic_only": True,
                **FLAGS,
            },
        )
        bind_identity(identities, HERE / "motions.json")
    for path in list(identities):
        bind_identity(identities, path)
    if hashes != {"source": compiled_model_sha256(source), "target": compiled_model_sha256(target)}:
        raise ValueError("task fitting mutated a compiled model")
    dump(
        HERE / "report.json",
        {
            "kind": "g1_true23_original_planner_se3_task_fit_corpus_v1",
            "records": records,
            "inputs": identities,
            "compiled_models": hashes,
            "config": asdict(config),
            "all_attempts_completed": True,
            "clips_written": sum(bool(r["output"]) for r in records),
            "source30_to_target50_time_scale": 1.0,
            "contact_force_qualification_performed": False,
            "controller_replay_or_training_performed": False,
            **FLAGS,
        },
    )


if __name__ == "__main__":
    main()
