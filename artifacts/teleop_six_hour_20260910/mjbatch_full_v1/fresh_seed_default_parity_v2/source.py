"""Compare pre-seed frozen runner against new default-off physical execution."""

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gear_sonic.scripts import evaluate_g1_true23_mjbatch_mpc as current
from gear_sonic.utils.g1_true23_mjbatch_mpc import sha256


def run(args):
    args.output.mkdir(exist_ok=False)
    old_root = args.output / "old" / "gear_sonic"
    old_script = old_root / "scripts" / "evaluate_g1_true23_mjbatch_mpc.py"
    old_script.parent.mkdir(parents=True)
    old_script.write_bytes(args.frozen_runner.read_bytes())
    old_utils = old_root / "utils"
    old_utils.mkdir()
    utility = Path(current.__file__).resolve().parents[1] / "utils"
    for name in (
        "g1_true23_mjbatch_ilqr_core.py", "g1_true23_mjbatch_model.py",
        "g1_true23_mjbatch_mpc.py", "g1_true23_relative_foot_cost.py",
    ):
        (old_utils / name).write_bytes((utility / name).read_bytes())
    spec = importlib.util.spec_from_file_location("pre_fresh_runner", old_script)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    namespace = dict(
        bundle=args.bundle, clip="pico", probe="standing", standing_seconds=0.08,
        source_seconds=3, horizon=2, commit=1, iterations=1, threads=1,
        feedback_clip=0.1, checkpoint_controls=0, motion_override=None,
        target_seed=args.target_seed, ankle_limit_margin=None, ankle_limit_weight=None,
        all_joint_limit_margin=None, all_joint_limit_weight=None, relative_foot_weight=0,
        fd_epsilon=1e-6, fresh_bfm_seed_onnx=None, fresh_bfm_seed_dependencies=None,
    )
    for name, module in (("old_run", old), ("new_run", current)):
        result = module.run(SimpleNamespace(**namespace, output=args.output / name))
        assert result["probe_completed"] and result["failure"] is None
    checks = {}
    with np.load(args.output / "old_run/trace.npz") as before:
        with np.load(args.output / "new_run/trace.npz") as after:
            assert set(before.files) == set(after.files)
            for key in before.files:
                np.testing.assert_array_equal(before[key], after[key], err_msg=key)
                checks[key] = dict(shape=list(before[key].shape), bit_exact=True)
    report = dict(
        kind="fresh_seed_disabled_physical_regression", controls=4, physics_substeps=40,
        every_trace_array_bit_exact=True, trace_checks=checks,
        old_runner_sha256=sha256(args.frozen_runner), new_runner_sha256=sha256(Path(current.__file__)),
        warning_and_clock_guards_quiet=True,
        scope="default-off physical and feedback trace; wall-time/provenance fields intentionally differ",
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    (args.output / "source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--frozen-runner", type=Path, required=True)
    parser.add_argument("--target-seed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
