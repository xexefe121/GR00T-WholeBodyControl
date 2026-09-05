"""Audit action-independent initial target infeasibility at corpus reset states.

No inference, training, robot or parameter mutation. This checks nominal
reference joint positions/velocities, not randomized resets or dynamic balance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_HARD_LOWER_HARDWARE,
    SAFE_TARGET_HARD_UPPER_HARDWARE,
    SAFE_TARGET_SOFT_LOWER_HARDWARE,
    SAFE_TARGET_SOFT_UPPER_HARDWARE,
)
from gear_sonic.utils.g1_true23_actuation_profile import NativeSupportActuationProfile


def reset_intersections(q, dq, profile):
    """Return feasibility with target=q versus any feasible previous PD target."""
    if (
        q.ndim != 2
        or q.shape[1] != 23
        or dq.shape != q.shape
        or not np.isfinite(q).all()
        or not np.isfinite(dq).all()
    ):
        raise ValueError("reset audit requires finite [N,23] q/dq")
    kp, kd, effort = (np.asarray(getattr(profile, name)) for name in ("kp", "kd", "effort"))
    cap = 0.95 * 0.25 * effort
    lower = np.maximum(np.asarray(SAFE_TARGET_HARD_LOWER_HARDWARE) + 0.05, q + (kd * dq - cap) / kp)
    upper = np.minimum(np.asarray(SAFE_TARGET_HARD_UPPER_HARDWARE) - 0.05, q + (kd * dq + cap) / kp)
    delta = profile.slew_rad_s * profile.timestep_s
    q_seed_infeasible = np.maximum(lower, q - delta) > np.minimum(upper, q + delta)
    any_seed_infeasible = lower > upper
    return q_seed_infeasible, any_seed_infeasible


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--spans", type=Path, required=True)
    parser.add_argument("--sim-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    profile = NativeSupportActuationProfile.from_sim_config(args.sim_config)
    sidecar = json.loads(args.spans.read_text())
    if sidecar.get("kind") != "g1_true23_motion_corpus_spans_v1":
        raise ValueError("reset audit requires causal corpus span metadata")
    spans = sidecar["spans"]
    with np.load(args.motion, allow_pickle=False) as archive:
        q, dq = archive["joint_pos"].copy(), archive["joint_vel"].copy()
        fps = np.asarray(archive["fps"])
    if fps.size != 1 or float(fps.reshape(-1)[0]) != 50.0:
        raise ValueError("reset audit requires 50 Hz motion")
    if len(spans) != sidecar["clip_count"] or len(q) != sidecar["total_frames"]:
        raise ValueError("reset audit corpus frame count mismatch")
    # Same nominal soft-position clipping used by the command reset. Random
    # pose/joint/velocity perturbations are deliberately absent from this audit.
    q = np.clip(q, SAFE_TARGET_SOFT_LOWER_HARDWARE, SAFE_TARGET_SOFT_UPPER_HARDWARE)
    clips = []
    previous_stop = 0
    for span in spans:
        start, stop = int(span["start"]), int(span["start"]) + int(span["length"])
        indexes = np.arange(start + 13, stop - 1)
        if start != previous_stop or stop > len(q) or len(indexes) == 0:
            raise ValueError("invalid causal corpus span")
        previous_stop = stop
        q_seed, any_seed = reset_intersections(q[indexes], dq[indexes], profile)
        clips.append(
            {
                "name": span["name"],
                "nominal_control_frames": len(indexes),
                "infeasible_when_previous_target_is_q": int(np.any(q_seed, axis=1).sum()),
                "infeasible_for_any_previous_target": int(np.any(any_seed, axis=1).sum()),
                "q_seed_infeasible_counts_by_joint": dict(
                    zip(HARDWARE_23_JOINT_NAMES, q_seed.sum(axis=0).tolist())
                ),
            }
        )
    if previous_stop != len(q):
        raise ValueError("reset audit spans do not cover full corpus")
    report = {
        "kind": "g1_true23_nominal_reset_target_feasibility_v1",
        "clips": clips,
        "profile": profile.contract(),
        "hardware_authorized": False,
        "robot_commands_published": False,
        "dynamic_feasibility_proven": False,
        "scope": "action_independent_nominal_reference_reset_intersections_not_randomization_or_dynamic_feasibility",
        "sources": {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (args.motion, args.spans, args.sim_config, Path(__file__))
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                "clips": [
                    {key: value for key, value in clip.items() if key != "q_seed_infeasible_counts_by_joint"}
                    for clip in clips
                ]
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
