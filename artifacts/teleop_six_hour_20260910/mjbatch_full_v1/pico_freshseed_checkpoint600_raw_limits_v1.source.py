"""Read-only all-substep native limit witnesses from a committed MPC trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle, sha256


def run(args):
    model, contract, _, timeline, _ = load_native_bundle(args.bundle, args.clip)
    phase = next(item for item in timeline["phases"] if item["name"] == "source_motion")
    with np.load(args.trace, allow_pickle=False) as archive:
        q = archive["physics_qpos"][:, 7:].copy()
        dq = archive["physics_qvel"][:, 6:].copy()
        torque = archive["physics_torque"].copy()
        substeps = archive["physics_substeps"].copy()
        targets = archive["target"].copy()
    bounds = model.jnt_range[1:]
    excess = np.maximum(np.maximum(bounds[:, 0] - q, q - bounds[:, 1]), 0)
    control_for_physics = np.repeat(np.arange(len(substeps)), substeps)
    assert len(control_for_physics) == len(q) - 1 == len(torque)
    witnesses = []
    for joint in np.flatnonzero(np.max(excess, axis=0) > 1e-6):
        samples = np.flatnonzero(excess[:, joint] > 1e-6)
        peak = int(np.argmax(excess[:, joint]))
        control = int(control_for_physics[max(peak - 1, 0)])
        witnesses.append(
            dict(
                joint=contract["joint_names"][joint],
                native_index=int(joint),
                violating_samples=len(samples),
                first_physics_sample=int(samples[0]),
                peak_physics_sample=peak,
                source_elapsed_seconds_at_peak=float(peak * 0.002 - phase["control_start"] * 0.02),
                peak_excess_rad=float(excess[peak, joint]),
                q_at_peak=float(q[peak, joint]),
                dq_at_peak=float(dq[peak, joint]),
                executed_target_at_peak=float(targets[control, joint]),
                native_bounds=bounds[joint].tolist(),
            )
        )
    report = dict(
        kind="independent_all_physics_native_limits_from_committed_trace",
        clip=args.clip,
        trace_sha256=sha256(args.trace),
        code_sha256=sha256(__file__),
        physics_steps=len(q) - 1,
        control_slots=len(targets),
        fully_completed_controls=int(np.sum(substeps == 10)),
        original_source_requested_controls=phase["requested_controls"],
        attempted_source_controls=max(0, min(len(targets), phase["control_stop"]) - phase["control_start"]),
        range_tolerance_rad=1e-6,
        range_excess_max_rad=float(excess.max()),
        speed_ratio_max=float(np.max(np.abs(dq) / np.asarray(contract["native_velocity"]))),
        effort_ratio_max=float(np.max(np.abs(torque) / np.asarray(contract["native_effort"]))),
        range_witnesses=witnesses,
        strict_native_limits_pass=bool(
            excess.max() <= 1e-6
            and np.max(np.abs(dq) / np.asarray(contract["native_velocity"])) <= 1 + 1e-8
            and np.max(np.abs(torque) / np.asarray(contract["native_effort"])) <= 1 + 1e-8
        ),
        full_body_tracking_qualification=False,
    )
    args.output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("bundle", "trace", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--clip", required=True)
    run(parser.parse_args())
