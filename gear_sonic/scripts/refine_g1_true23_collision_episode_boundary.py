"""One bounded warm-start refinement; no full reference promotion."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from gear_sonic.utils.g1_true23_mpc_student import load_inputs, sha256
from gear_sonic.utils.g1_true23_arm_collision_preview import TwoFrameArmProjection


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    model, contract, _, original, _ = load_inputs(args.bundle, "pico")
    reference = args.references / "pico/reference.npz"
    with np.load(reference, allow_pickle=False) as a:
        motion = {k: a[k].copy() for k in a.files}
    with np.load(args.previous / "episode.npz", allow_pickle=False) as a:
        frame = int(a["frames"][-1])
        previous = a["proposed_current"][-2].copy()
        seed = np.array([a["proposed_current"][-1, 20:], a["proposed_next"][-1, 20:]])
    poses = np.array([np.r_[motion["body_pos_w"][f, 0], motion["body_quat_w"][f, 0],
                             motion["joint_pos"][f]] for f in (frame, frame + 1)])
    goals = np.array([poses[i, :3] + original["source_task_position_w"][f, :2]
                      - original["source_qpos29"][f, :3] for i, f in enumerate((frame, frame + 1))])
    projection = TwoFrameArmProjection(model, .8 * np.asarray(contract["native_velocity"])[13:] * .02,
                                       max_iterations=args.iterations)
    request = dict(frame=frame, reference_sha256=sha256(reference),
                   previous_episode_sha256=sha256(args.previous / "episode.npz"),
                   maximum_iterations=args.iterations, source_clock_hz=50, explicit_preview_seconds=.02,
                   physical_hand_gate_m=.15, optimization_hand_margin_m=.149,
                   no_physical_rollout=True, full_reference_promoted=False)
    (args.output / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    (args.output / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    (args.output / "projector_snapshot.py").write_bytes(
        (Path(__file__).resolve().parents[1] / "utils/g1_true23_arm_collision_preview.py").read_bytes())
    started = time.perf_counter()
    proposed, report = projection.solve(poses, goals, previous, seed=seed)
    report.update(frame=frame, elapsed_s=time.perf_counter() - started, no_physical_rollout=True,
                  full_reference_promoted=False)
    np.savez_compressed(args.output / "poses.npz", original=poses, previous=previous,
                        initial_seed=seed, proposed=proposed, goals=goals)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("bundle", "references", "previous", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--iterations", type=int, default=180)
    run(p.parse_args())
