"""Compare new CPU replay features against actual saved GPU learner states."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from gear_sonic.utils.g1_true23_compact_replay import CompactAdapter, CompactPolicy
from gear_sonic.utils.g1_true23_compact_features import pack_observation
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    request = json.loads((args.run / "request.json").read_text())
    bank = json.loads(Path(request["bank_report"]).read_text())
    spec = json.loads(Path(bank["files"]["spec"]).read_text())
    with np.load(spec["files"]["native_motion"]["path"], allow_pickle=False) as z:
        motion = {k: z[k].copy() for k in z.files}
    with np.load(spec["files"]["original_reference"]["path"], allow_pickle=False) as z:
        vr = z["virtual_vr21"].copy()
    with np.load(args.run / "sampled_inputs_rewards.npz", allow_pickle=False) as z:
        samples = {k: z[k].copy() for k in z.files}
    model = CompactAdapter(motion, vr, root=ROOT, assets=ROOT.parent / "GR00T-WholeBodyControl")
    outcome = json.loads((args.run / "outcome.json").read_text())
    policy = CompactPolicy(args.run / f"compact_{outcome['updates']:04d}.pt")
    maximum, count = 0.0, 0
    features = []
    for t in range(len(samples["update"])):
        for e in range(samples["policy"].shape[1]):
            qpos = np.r_[
                samples["root_position_w"][t, e] - samples["env_origins"][t, e],
                samples["root_quaternion_wxyz"][t, e],
                samples["qpos_hw"][t, e],
            ].astype(np.float64)
            qvel = np.r_[samples["root_velocity_w"][t, e], np.zeros(3), samples["dqpos_hw"][t, e]].astype(
                np.float64
            )
            goal = model.goal_from_copies(int(samples["reference_q0"][t, e]) - 9, qpos, qvel)
            maximum = max(maximum, float(np.max(np.abs(goal - samples["native_goal"][t, e]))))
            with torch.inference_mode():
                value = (
                    pack_observation(
                        torch.from_numpy(samples["policy"][t, e][None]), torch.from_numpy(goal[None])
                    )[0]
                    .numpy()
                    .copy()
                )
            features.append(value)
            count += 1
    if maximum > 1e-5:
        raise ValueError("CPU replay differs from actual GPU learner features: " + str(maximum))
    # Repeated CPU timing is a neural-only diagnostic, NOT whole-loop deadline qualification.
    elapsed = []
    for _ in range(10):
        for value in features:
            start = time.perf_counter()
            action = policy.predict(value)
            elapsed.append(time.perf_counter() - start)
            if action.shape != (23,) or not np.isfinite(action).all() or np.max(np.abs(action)) >= 10:
                raise ValueError("compact CPU inference violated existing action ABI")
    result = dict(
        passed=True,
        actual_training_states=count,
        cpu_goal_max_abs_error=maximum,
        neural_calls=len(elapsed),
        neural_only_ms_p50_p95_max=(1000 * np.quantile(elapsed, [0.5, 0.95, 1])).tolist(),
        source_hashes={str(Path(__file__)): sha256_file(Path(__file__))},
        checkpoint=policy.identity(),
        deployment_ready=False,
        hardware_authorized=False,
    )
    with (args.run / "replay_boundary_audit.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
