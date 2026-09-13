"""Bounded two-planner committed-RAM probe; no long physical replay."""

import argparse
import json
from pathlib import Path
import resource

import mujoco
import numpy as np

from gear_sonic.utils.g1_true23_mjbatch_ilqr_core import ilqr
from gear_sonic.utils.g1_true23_mjbatch_model import position_servo_copy
from gear_sonic.utils.g1_true23_mjbatch_mpc import Native23Tracker, load_native_bundle, sha256


def memory(path):
    return {line.split(":", 1)[0]: line.split(":", 1)[1].strip()
            for line in Path(path).read_text().splitlines() if ":" in line}


def run(args):
    native, contract, motion, _, _ = load_native_bundle(args.bundle, "pico")
    before = memory("/proc/meminfo")
    planners, records = [], []
    for index in range(2):
        model = position_servo_copy(native, contract["kp"], contract["kd"], contract["native_effort"])
        planner = Native23Tracker(model, contract, motion, horizon=30, threads=4)
        ilqr(planner, planner.states[10], planner.target_reference(np.arange(30)), iters=1)
        planners.append(planner)
        status = memory("/proc/self/status")
        records.append(dict(live_planners=index + 1, VmRSS=status["VmRSS"], VmHWM=status["VmHWM"],
                            VmSize=status["VmSize"], available=memory("/proc/meminfo")["MemAvailable"]))
    report = dict(kind="two_native23_H30_four_thread_planners_committed_ram_probe",
                  mujoco=mujoco.__version__, records=records, initial_memory=before,
                  maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  code_sha256=sha256(__file__),
                  limitation="Two planners in one process; separate processes add another interpreter/import overhead.")
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "initial_memory"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
