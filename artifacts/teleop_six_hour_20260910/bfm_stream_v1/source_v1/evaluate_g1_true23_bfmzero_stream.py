"""Exercise received-only BFM replay and faults on the native23 simulator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from gear_sonic.scripts.evaluate_g1_true23_bfmzero import DATA, MODEL, PACKAGE, PHYSICS, ROOT, load_motion
from gear_sonic.utils.g1_true23_bfmzero_inference import BFMZeroInference, load_contract
from gear_sonic.utils.g1_true23_bfmzero_stream import BFMStreamSimulator, DT, Packet, packet_fields
from gear_sonic.utils.g1_true23_step1b_mujoco import _quaternion_matrix, prepare_true23_model


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def delivery_schedule(count, scenario, fault_at, pause_seconds, rearm_at, restart_seconds):
    """Return source events, without generating payloads until receiver delivery."""
    events = []
    fault_sequence = math.ceil(fault_at / DT - 1e-9)
    for sequence in range(count):
        arrival = sequence * DT
        if scenario in ("disconnect", "resume") and sequence >= fault_sequence:
            continue
        if scenario == "pause" and fault_at <= arrival < fault_at + pause_seconds - 1e-9:
            continue
        if scenario == "reorder":
            if sequence == fault_sequence:
                arrival += DT
            elif sequence == fault_sequence + 1:
                arrival -= DT
        corrupt = scenario == "nonfinite" and sequence == fault_sequence
        events.append((arrival, 1, 0, sequence, corrupt, sequence == count - 1))
    if scenario == "resume":
        events.append((rearm_at, 0, 1, -1, False, False))
        restart_count = min(count, math.ceil(restart_seconds / DT))
        for sequence in range(restart_count):
            events.append((rearm_at + sequence * DT, 1, 1, sequence, False, sequence == count - 1))
    return sorted(events)


def run(args):
    if args.output.exists():
        raise FileExistsError("stream evidence refuses overwrite")
    if (
        args.fault_at < 0
        or args.pause_seconds <= 0.1
        or args.stop_seconds <= 0
        or args.restart_seconds <= 0
        or (args.max_seconds is not None and args.max_seconds <= 0)
    ):
        raise ValueError("invalid stream experiment timing")
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True)
    config_path = PACKAGE / "bfmzero_inspect_v1/config.yaml"
    weights_path = PACKAGE / "bfmzero_inference_v1/inference.safetensors"
    model_path = ROOT.parent / "GR00T-WholeBodyControl" / MODEL
    physics_path = ROOT / PHYSICS
    contract = load_contract(config_path)
    policy = BFMZeroInference(weights_path)
    _, model, physics = prepare_true23_model(model_path, physics_path)
    physical_config = json.loads(physics_path.read_text())
    motion, timeline, source_path = load_motion(args.clip)
    expected_source_sha = timeline.get("timeline_sha256")
    if expected_source_sha is None or sha256(source_path) != expected_source_sha:
        raise ValueError("saved stream timeline does not match bound source bytes")
    count = len(motion["joint_pos"]) - 11
    expected_bodies = tuple(model.body(index).name for index in range(1, model.nbody))
    if expected_bodies != contract["body_names"]:
        raise ValueError("stream source/model body order mismatch")
    initial_qpos = np.r_[motion["body_pos_w"][10, 0], motion["body_quat_w"][10, 0], motion["joint_pos"][10]]
    initial_qvel = np.r_[
        motion["body_lin_vel_w"][10, 0],
        _quaternion_matrix(initial_qpos[3:7]).T @ motion["body_ang_vel_w"][10, 0],
        motion["joint_vel"][10],
    ]
    standing = physical_config["initial_state"]
    simulator = BFMStreamSimulator(
        model,
        physics,
        contract,
        policy,
        initial_qpos,
        initial_qvel,
        np.asarray(standing["joint_position_hardware_rad"]),
        standing["base_position_m"][2],
        np.asarray(physical_config["physics"]["velocity_limit_hardware_radps"]),
        position_gain=args.position_gain,
        yaw_gain=args.yaw_gain,
        stop_seconds=args.stop_seconds,
    )
    rearm_at = args.rearm_at if args.rearm_at is not None else args.fault_at + 6.0
    if args.scenario == "resume" and rearm_at <= args.fault_at + args.stop_seconds + 1:
        raise ValueError("explicit rearm must allow stop plus standing verification")
    schedule = delivery_schedule(
        count, args.scenario, args.fault_at, args.pause_seconds, rearm_at, args.restart_seconds
    )
    if args.max_seconds is not None:
        duration = args.max_seconds
    elif args.scenario == "normal":
        duration = count * DT + 0.14 + 8.0
    elif args.scenario == "resume":
        duration = rearm_at + args.restart_seconds + 8.0
    else:
        duration = args.fault_at + 9.0
    requested_controls = math.ceil(duration / DT - 1e-9)
    request = {
        "scenario": args.scenario,
        "clip": args.clip,
        "source_packets": count,
        "requested_controls": requested_controls,
        "fault_at": args.fault_at,
        "pause_seconds": args.pause_seconds,
        "rearm_at": rearm_at,
        "restart_seconds": args.restart_seconds,
        "paced": args.paced,
        "position_gain": args.position_gain,
        "yaw_gain": args.yaw_gain,
        "source_timeline": timeline,
        "source_report_root": str(DATA),
        "hardware_authorized": False,
        "inputs": {
            str(path): sha256(path)
            for path in (
                source_path,
                model_path,
                physics_path,
                config_path,
                weights_path,
                Path(__file__),
                ROOT / "gear_sonic/utils/g1_true23_bfmzero_stream.py",
                ROOT / "gear_sonic/utils/g1_true23_bfmzero_inference.py",
                ROOT / "gear_sonic/scripts/evaluate_g1_true23_bfmzero.py",
            )
        },
    }
    (args.output / "request.json").write_text(json.dumps(request, indent=2, allow_nan=False))
    durations, lateness, deadline_misses = [], [], []
    cursor = 0
    rearm_events = []
    # Warm only the CPU kernels; no simulator transition or action history change.
    zero = torch.zeros
    policy.actor(
        zero(1, 52), zero(1, 23), zero(1, 300), torch.nn.functional.normalize(torch.ones(1, 256), dim=-1) * 16
    )
    wall_start = time.perf_counter()
    for control in range(requested_controls):
        scheduled_time = control * DT
        if args.paced:
            remaining = wall_start + scheduled_time - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
        tick = time.perf_counter()
        now = tick - wall_start if args.paced else scheduled_time
        while cursor < len(schedule) and schedule[cursor][0] <= now + 1e-7:
            delivery_time, kind, epoch, sequence, corrupt, final = schedule[cursor]
            cursor += 1
            if kind == 0:
                accepted = simulator.rearm(delivery_time)
                rearm_events.append({"scheduled_time": delivery_time, "received_time": now, "accepted": accepted})
                continue
            fields = packet_fields(motion, sequence + 11)
            if corrupt:
                fields["joint_pos"][3] = np.nan
            simulator.receive(Packet(epoch, sequence, sequence * DT, fields, final), now)
        simulator.step(now)
        ended = time.perf_counter()
        durations.append((ended - tick) * 1000)
        lateness.append(max(0.0, (tick - wall_start - scheduled_time) * 1000) if args.paced else 0.0)
        deadline_misses.append(ended > wall_start + (control + 1) * DT if args.paced else ended - tick > DT)
        if simulator.physical_failure is not None:
            break
        if (control + 1) % 500 == 0:
            print(
                json.dumps(
                    {
                        "controls": control + 1,
                        "mode": simulator.mode.value,
                        "source_consumed": simulator.gate.consumed,
                        "epoch": simulator.gate.epoch,
                    }
                ),
                flush=True,
            )
    wall_elapsed = time.perf_counter() - wall_start
    arrays = simulator.arrays()
    arrays.update(
        full_loop_ms=np.asarray(durations),
        control_start_late_ms=np.asarray(lateness),
        deadline_missed=np.asarray(deadline_misses),
        arrivals=np.asarray(simulator.arrivals),
    )
    with (args.output / "trace.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    result = simulator.report()
    result.update(
        {
            "scenario": args.scenario,
            "clip": args.clip,
            "paced": args.paced,
            "requested_controls": requested_controls,
            "budget_completed": simulator.controls == requested_controls,
            "source_packets_requested": count,
            "rearm_events": rearm_events,
            "wall_elapsed_seconds": wall_elapsed,
            "simulated_seconds": simulator.controls * DT,
            "full_loop_ms_p50_p95_max": np.percentile(durations, [50, 95, 100]).tolist(),
            "control_start_late_ms_p95_max": np.percentile(lateness, [95, 100]).tolist(),
            "deadline_misses": int(np.count_nonzero(deadline_misses)),
            "timing_qualified": args.paced and not any(deadline_misses),
            "source_tracking_qualified": False,
            "request_sha256": sha256(args.output / "request.json"),
            "trace_sha256": sha256(args.output / "trace.npz"),
        }
    )
    is_fault = args.scenario != "normal"
    protocol_pass = result["any_fault_latched"] if is_fault else result["full_uninterrupted_source_consumed"]
    if args.scenario == "resume":
        consumed_after_rearm = int(
            np.count_nonzero((arrays["source_epoch"] == 1) & (arrays["source_sequence"] >= 350))
        )
        result["source_motion_controls_after_explicit_rearm"] = consumed_after_rearm
        protocol_pass &= any(row["accepted"] for row in rearm_events) and consumed_after_rearm > 0
    result["protocol_scenario_passed"] = bool(protocol_pass)
    result["physical_lifecycle_passed"] = (
        simulator.physical_failure is None
        and result["standing_return_verified"]
        and result["range_excess_max_rad"] <= 1e-8
        and result["effort_ratio_max"] <= 1 + 1e-8
        and result["velocity_ratio_max"] <= 1
    )
    result["scenario_passed"] = result["protocol_scenario_passed"] and result["physical_lifecycle_passed"]
    (args.output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({key: value for key, value in result.items() if key != "events"}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", choices=("pico", "walk002", "walk003", "walk008"), default="walk002")
    parser.add_argument(
        "--scenario", choices=("normal", "pause", "disconnect", "reorder", "nonfinite", "resume"), default="normal"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fault-at", type=float, default=9.0)
    parser.add_argument("--pause-seconds", type=float, default=0.5)
    parser.add_argument("--rearm-at", type=float)
    parser.add_argument("--restart-seconds", type=float, default=8.5)
    parser.add_argument("--stop-seconds", type=float, default=2.0)
    parser.add_argument("--position-gain", type=float, default=1.0)
    parser.add_argument("--yaw-gain", type=float, default=2.0)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--paced", action="store_true")
    run(parser.parse_args())
