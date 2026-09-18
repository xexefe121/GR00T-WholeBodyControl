"""Verify allocation-only IMU odometry changes against the legacy hot path."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from gear_sonic.scripts.evaluate_g1_true23_bfmzero import PHYSICS, ROOT
from gear_sonic.utils.g1_true23_bfm_imu_odometry import Native23IMUOdometry
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model


def main(args):
    samples = np.load(args.sensor_samples)
    required = (
        "timestamp_s", "joint_q", "joint_dq", "imu_quat_wxyz", "gyro_body",
        "accel_specific_force_body",
    )
    if any(name not in samples for name in required):
        raise ValueError("sensor archive has incomplete odometry inputs")
    if args.samples <= 0 or len(samples["timestamp_s"]) < args.samples:
        raise ValueError("sensor archive has too few samples")
    _, model, _ = prepare_true23_model(ROOT.parent / "GR00T-WholeBodyControl" / MODEL, ROOT / PHYSICS)
    optimized = Native23IMUOdometry(model)
    legacy = Native23IMUOdometry(model)
    outputs = ("position_start", "velocity_start", "accel_bias_body", "quaternion_start", "support_weights")
    maximum = {name: 0.0 for name in outputs}
    optimized_seconds = legacy_seconds = 0.0
    for index in range(args.samples):
        packet = {name: samples[name][index] for name in required}
        started = time.perf_counter()
        left = optimized.update(**packet)
        optimized_seconds += time.perf_counter() - started
        started = time.perf_counter()
        right = legacy.update(**packet)
        legacy_seconds += time.perf_counter() - started
        for name in outputs:
            maximum[name] = max(maximum[name], float(np.max(np.abs(left[name] - right[name]))))
        if left["timestamp_s"] != right["timestamp_s"]:
            raise AssertionError("timestamp mismatch")
    result = {
        "samples": args.samples,
        "max_absolute_difference": maximum,
        "overall_max_absolute_difference": max(maximum.values()),
        "threshold": 1e-9,
        "passed": max(maximum.values()) < 1e-9,
        "mean_update_ms": {
            "optimized": optimized_seconds * 1000.0 / args.samples,
            "legacy": legacy_seconds * 1000.0 / args.samples,
            "saving": (legacy_seconds - optimized_seconds) * 1000.0 / args.samples,
        },
    }
    if not result["passed"]:
        raise AssertionError(json.dumps(result))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor-samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2000)
    main(parser.parse_args())
