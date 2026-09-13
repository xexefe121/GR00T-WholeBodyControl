"""Read-only learning diagnosis, not checkpoint selection or motion acceptance."""

import argparse
import json
from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file


def distribution(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    assert len(values) and np.isfinite(values).all()
    return dict(
        count=len(values),
        mean=float(values.mean()),
        p50=float(np.median(values)),
        p95=float(np.percentile(values, 95)),
        maximum=float(values.max()),
    )


def main(directory):
    output = directory / "learning_curve.json"
    if output.exists():
        raise FileExistsError("learning diagnosis refuses overwrite")
    pins = {str(Path(__file__).resolve()): sha256_file(Path(__file__))}

    def bind(path):
        path = Path(path).resolve(strict=True)
        pins[str(path)] = sha256_file(path)
        return path

    def read(name):
        return json.loads(bind(directory / name).read_text())

    request, outcome = read("request.json"), read("outcome.json")
    assert outcome["completed"] and outcome["simulator_updates"] == 500
    args = request["arguments"]
    spans = json.loads(bind(args["spans"]).read_text())["spans"]
    with np.load(bind(directory / "reward_capture.npz"), allow_pickle=False) as z:
        costs = z["world_cost_parts_before"].copy()
        done = z["stored_done"].copy()
        term, timeout = z["terminated"].copy(), z["timeouts"].copy()
        bonus, reward = z["world_quality_bonus"].copy(), z["returned_reward"].copy()
    with np.load(bind(directory / "sampled_actual_inputs.npz"), allow_pickle=False) as z:
        controls, anchors = z["control_index"].copy(), z["reference_q0"].copy()
    assert costs.shape == (8000, 128, 3)
    assert done.dtype == bool and done.shape == (8000, 128)
    np.testing.assert_array_equal(done, term | timeout)
    assert costs.min() >= 0 and np.isfinite(costs).all()
    # Contract: parts[0] = 10*root_error^2; parts[1] = mean(feet_error^2)/0.05^2.
    # These are pre-action current-setpoint errors, not p95 individual-foot gates.
    roots = np.sqrt(costs[..., 0] / 10)
    feet_rms = 0.05 * np.sqrt(costs[..., 1])
    quality = 1 / (1 + costs.sum(-1))
    starts = np.zeros(128, dtype=np.int64)
    episodes = []
    for control, env in zip(*np.nonzero(done), strict=True):
        episodes.append((int(control), int(env), int(control - starts[env] + 1)))
        starts[env] = control + 1
    assert sum(row[2] for row in episodes) + int((8000 - starts).sum()) == 1_024_000
    windows = []
    for begin in range(0, 8000, 1600):
        end = begin + 1600
        sl = slice(begin, end)
        lengths = [length for control, _, length in episodes if begin <= control < end]
        sampled = (controls >= begin) & (controls < end)
        rows = []
        sample_costs = costs[controls[sampled]]
        q1 = anchors[sampled] + 1
        covered = np.zeros_like(q1, dtype=bool)
        for span in spans:
            in_clip = (q1 >= span["start"]) & (q1 < span["start"] + span["length"])
            assert not np.any(covered & in_clip)
            covered |= in_clip
            source = next(p for p in span["timeline"]["phases"] if p["name"] == "source_motion")
            in_source = (
                in_clip
                & (q1 >= span["start"] + source["frame_start"])
                & (q1 < span["start"] + source["frame_stop"])
            )
            values = sample_costs[in_source]
            row = dict(
                name=span["name"],
                sampled_clip_env_controls=int(in_clip.sum()),
                sampled_source_env_controls=int(in_source.sum()),
            )
            if len(values):
                row.update(
                    root_error_m=distribution(np.sqrt(values[:, 0] / 10)),
                    feet_world_rms_m=distribution(0.05 * np.sqrt(values[:, 1])),
                    joint_world_quality=distribution(1 / (1 + values.sum(-1))),
                )
            rows.append(row)
        assert covered.all()
        windows.append(
            dict(
                updates=[begin // 16 + 1, end // 16],
                transitions=1600 * 128,
                termination_fraction=float(term[sl].mean()),
                timeout_fraction=float(timeout[sl].mean()),
                completed_reset_intervals_controls=distribution(lengths),
                root_error_m=distribution(roots[sl]),
                feet_world_rms_m=distribution(feet_rms[sl]),
                joint_world_quality=distribution(quality[sl]),
                returned_reward=distribution(reward[sl]),
                non_done_quality_bonus_mean=float(bonus[sl][~done[sl]].mean()),
                sampled_source_by_clip=rows,
            )
        )
    report = dict(
        kind="existing_pico_training_learning_curve_v1",
        inputs=pins,
        windows=windows,
        completed_reset_intervals=len(episodes),
        final_censored_reset_interval_controls=distribution(8000 - starts),
        sampled_phase="pre_action_received_q1_setpoint",
        nonstationary_reset_distribution_comparison=True,
        reset_interval_is_complete_motion=False,
        sampled_coverage_is_full_exposure=False,
        policy_tracking_or_generalization_pass_claimed=False,
        deployment_ready=False,
        hardware_authorized=False,
    )
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    for window in windows:
        print(json.dumps(window), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    main(parser.parse_args().directory.resolve(strict=True))
