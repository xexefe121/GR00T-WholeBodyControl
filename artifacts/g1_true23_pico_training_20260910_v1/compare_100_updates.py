"""Same-budget/same-prefix comparison with prior PICO-excluded100-update run."""

import json
from pathlib import Path

import numpy as np

from evaluate_checkpoint import metrics
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file

HERE = Path(__file__).resolve().parent
OLD = Path("/mnt/c/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/normal_core_adaptation_v1")


def main():
    output = HERE / "eval100_v1/equal_budget_comparison.json"
    if output.exists():
        raise FileExistsError("equal-budget comparison refuses overwrite")
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = sha256_file(path)
        if expected is not None:
            assert digest == expected, path
        inputs[str(path)] = digest
        return path

    def read(path, expected=None):
        with np.load(bind(path, expected), allow_pickle=False) as z:
            return {k: z[k].copy() for k in z.files}

    bind(__file__)
    bind(HERE / "evaluate_checkpoint.py")
    original_config = json.loads(bind(OLD / "train100_v1/resolved_training.json").read_text())
    current_config = json.loads(bind(HERE / "train500_v1/resolved_training.json").read_text())
    assert original_config["pico_used_for_training"] is False and current_config["pico_used_for_training"] is True
    assert original_config["normal_adaptation"] == current_config["normal_adaptation"]
    old_agent, new_agent = original_config["agent"], current_config["agent"]
    differing = {k for k in old_agent if old_agent[k] != new_agent[k]}
    assert differing == {"max_iterations", "save_interval"}
    assert old_agent["num_steps_per_env"] == new_agent["num_steps_per_env"] == 16
    rows = []
    for name in ("walk002", "walk003", "walk008", "pico"):
        old_path, new_path = OLD / "cpu100_v2" / name, HERE / "eval100_v1" / name
        a = json.loads(bind(old_path / "report.json").read_text())
        b = json.loads(bind(new_path / "report.json").read_text())
        assert b["identity"]["completed_training_updates"] == 100
        assert a["timeline"] == b["timeline"]
        motion = read(a["timeline"]["timeline_path"], a["timeline"]["timeline_sha256"])
        ta = read(old_path / "trace.npz", a["inputs"][str(old_path / "trace.npz")])
        tb = read(new_path / "trace.npz", b["trace_sha256"])
        np.testing.assert_array_equal(ta["qpos"][0], tb["qpos"][0])
        common = min(len(ta["qpos"]), len(tb["qpos"])) - 1
        phase = next(r for r in a["timeline"]["phases"] if r["name"] == "source_motion")
        row = dict(
            name=name,
            common_controls=common,
            old_without_pico=metrics(ta, motion, phase, common),
            new_with_pico=metrics(tb, motion, phase, common),
            old_complete_source_controls=min(len(ta["qpos"]) - 1, phase["control_stop"]) - phase["control_start"],
            new_complete_source_controls=min(len(tb["qpos"]) - 1, phase["control_stop"]) - phase["control_start"],
            requested_source_controls=phase["requested_controls"],
            deployment_ready=False,
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    with output.open("x") as stream:
        json.dump(
            dict(
                cases=rows,
                same_completed_100_update_budget=True,
                same_actor_reward_and_optimizer_contract=True,
                data_membership_changed=True,
                wall_clock_runtime_identity_claimed=False,
                inputs=inputs,
                deployment_ready=False,
            ),
            stream,
            indent=2,
            allow_nan=False,
        )


if __name__ == "__main__":
    main()
