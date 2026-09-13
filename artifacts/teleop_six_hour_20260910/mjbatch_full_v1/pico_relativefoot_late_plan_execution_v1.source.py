"""Compare committed nominal states with actual pre-action states, without replanning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle, sha256


def run(args):
    request_path, trace_path, plans_path = [args.run / name for name in ("request.json", "trace.npz", "plans.json")]
    request = json.loads(request_path.read_text())
    plans = json.loads(plans_path.read_text())
    native, _, _, timeline, _ = load_native_bundle(args.bundle, request["clip"])
    phase = next(item for item in timeline["phases"] if item["name"] == "source_motion")
    with np.load(trace_path, allow_pickle=False) as archive:
        saved = {name: archive[name].copy() for name in (
            "qpos", "qvel", "planned_state", "planned_target", "target", "feedback_correction_raw",
            "feedback_correction_applied", "feedback_gain", "physics_substeps",
        )}
    count = len(saved["target"])
    assert saved["qpos"].shape == (count + 1, 30)
    tangent = np.empty((count, 58))
    for control in range(count):
        mujoco.mj_differentiatePos(
            native, tangent[control, :29], 1.0, saved["planned_state"][control, :30], saved["qpos"][control]
        )
    tangent[:, 29:] = saved["qvel"][:-1] - saved["planned_state"][:, 30:]
    raw_max = np.max(np.abs(saved["feedback_correction_raw"]), axis=1)
    first = max(0, min(args.late_start, count - 1))
    summaries = {}
    for name, mask in (
        ("all", np.ones(count, bool)),
        ("late_window", np.arange(count) >= first),
        ("planning_initial_states", np.arange(count) % request["commit"] == 0),
    ):
        current = tangent[mask]
        summaries[name] = dict(
            controls=int(mask.sum()),
            root_position_error_p95_max_m=np.percentile(np.linalg.norm(current[:, :3], axis=1), [95, 100]).tolist(),
            root_rotation_error_p95_max_rad=np.percentile(
                np.linalg.norm(current[:, 3:6], axis=1), [95, 100]
            ).tolist(),
            generalized_velocity_error_p95_max=np.percentile(np.max(np.abs(current[:, 29:]), axis=1), [95, 100]).tolist(),
            raw_feedback_correction_max_rad=float(raw_max[mask].max()),
        )
    report = dict(
        kind="committed_nominal_vs_actual_pre_action_state_audit",
        trace_sha256=sha256(trace_path),
        request_sha256=sha256(request_path),
        plans_sha256=sha256(plans_path),
        audit_source_sha256=sha256(__file__),
        mujoco=mujoco.__version__,
        controls=count,
        physics_steps=int(saved["physics_substeps"].sum()),
        late_start_control_zero_based=first,
        late_start_source_seconds=(first - phase["control_start"]) * 0.02,
        summaries=summaries,
        feedback_clipped_controls=np.flatnonzero(raw_max > request["feedback_correction_clip_rad"]).tolist(),
        late_plans=[plan for plan in plans if plan["control"] >= first],
        interpretation=(
            "Only committed nominal prefixes were stored. This measures following error about each local plan; "
            "it does not reconstruct an unstored full-H accepted trajectory or establish perturbation robustness. "
            "Seed costs are counterfactual rollouts from the current measured planning state."
        ),
    )
    args.output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("bundle", "run", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--late-start", type=int, default=810)
    run(parser.parse_args())
