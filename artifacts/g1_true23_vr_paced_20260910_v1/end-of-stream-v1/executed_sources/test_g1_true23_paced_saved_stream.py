"""Actual localhost publisher/consumer test using existing recorded motion.

One publisher thread owns its XPUB socket. Subscription acknowledgement avoids
silently omitting the beginning of the recording. The controller, actual 50-Hz
consumer, source timestamps and receiver timestamps share the WSL clock domain.
No external destination, XR device, robot channel or simulation pose animation.
"""

import argparse
import hashlib
import json
from pathlib import Path
import threading
import time

import numpy as np
import zmq

from gear_sonic.scripts.record_g1_true23_saved_teleop_diagnostic import build_recording_controller
from gear_sonic.scripts.replay_g1_true23_pico_packets_zmq import (
    load_reference_packets,
    rebase_reference_packet_time,
)
from gear_sonic.scripts.run_g1_true23_paced_sim import run_stream, save_result
from gear_sonic.teleop.paced_sim_runtime import LoopbackSubInbox
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import sha256_file
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
from gear_sonic.utils.g1_true23_generalist_source_closure import collect_local_source_closure

ROOT = Path(__file__).resolve().parents[2]


class SavedPublisher:
    def __init__(self, context, packets, *, scenario, fault_control, pause_controls):
        if scenario not in ("end-of-stream", "pause", "gap", "payload"):
            raise ValueError("unknown saved-source transport scenario")
        if not 2 <= len(packets) <= 10000 or not 2 <= fault_control < len(packets):
            raise ValueError("fault must follow startup within a bounded full source")
        if not 1 <= pause_controls <= len(packets) - fault_control:
            raise ValueError("pause must stay within the source")
        self.context, self.packets = context, packets
        self.scenario, self.fault_control, self.pause_controls = scenario, fault_control, pause_controls
        self.ready, self.stop = threading.Event(), threading.Event()
        self.endpoint, self.error, self.origin_ns = None, None, None
        self.sent = []
        self.thread = threading.Thread(target=self._run, name="recorded-reference-localhost-publisher")

    def _run(self):
        socket = self.context.socket(zmq.XPUB)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDHWM, 8)
        try:
            port = socket.bind_to_random_port("tcp://127.0.0.1")
            self.endpoint = f"tcp://127.0.0.1:{port}"
            self.ready.set()
            until = time.monotonic_ns() + 5_000_000_000
            while not self.stop.is_set():
                if socket.poll(20, zmq.POLLIN) and socket.recv() == b"\x01":
                    break
                if time.monotonic_ns() > until:
                    raise RuntimeError("no localhost subscriber acknowledgement")
            self.origin_ns = time.monotonic_ns() + 200_000_000
            for index, source in enumerate(self.packets):
                planned = self.origin_ns + index * 20_000_000
                while not self.stop.is_set() and time.monotonic_ns() < planned:
                    self.stop.wait(min((planned - time.monotonic_ns()) / 1e9, 0.02))
                if self.stop.is_set():
                    break
                if (self.scenario == "gap" and index == self.fault_control) or (
                    self.scenario == "pause"
                    and self.fault_control <= index < self.fault_control + self.pause_controls
                ):
                    continue
                packet = rebase_reference_packet_time(
                    source,
                    first_control_index=self.packets[0]["control_source_frame_index"],
                    first_control_monotonic_ns=self.origin_ns,
                )
                raw = json.dumps(packet, separators=(",", ":"), allow_nan=False).encode()
                if self.scenario == "payload" and index == self.fault_control:
                    raw = b"{deliberately malformed saved-input test"
                socket.send(raw, flags=zmq.NOBLOCK)
                self.sent.append(
                    dict(
                        source_index=source["control_source_frame_index"],
                        scheduled_ns=planned,
                        sent_ns=time.monotonic_ns(),
                        sha256=hashlib.sha256(raw).hexdigest(),
                    )
                )
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"
            self.ready.set()
        finally:
            socket.close()

    def start(self):
        self.thread.start()
        if not self.ready.wait(5) or self.endpoint is None:
            self.close()
            raise RuntimeError(self.error or "local publisher did not initialize")

    def close(self):
        self.stop.set()
        self.thread.join(2)
        if self.thread.is_alive():
            raise RuntimeError("local publisher failed to terminate")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository-root", "encoder-report", "decoder-report", "packets", "output-directory"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--scenario", choices=("end-of-stream", "pause", "gap", "payload"), required=True)
    parser.add_argument("--fault-control", type=int, default=200)
    parser.add_argument("--pause-controls", type=int, default=25)
    parser.add_argument("--tail-controls", type=int, default=250)
    parser.add_argument("--baseline-trace", type=Path)
    args = parser.parse_args(argv)
    args.legacy_unpaired_diagnostic = False
    # Source validation and expensive model/hash initialization precede streaming.
    packets = load_reference_packets(args.packets)
    controller, identity = build_recording_controller(args)
    materials = {
        str(path): sha256_file(path)
        for path in collect_local_source_closure(ROOT, [Path(__file__)]).as_source_files(ROOT).values()
    }
    materials.update(
        {
            str(args.packets.resolve()): sha256_file(args.packets),
            str(args.repository_root / MODEL): sha256_file(args.repository_root / MODEL),
            str(args.repository_root / PHYSICS): sha256_file(args.repository_root / PHYSICS),
        }
    )
    args.output_directory.mkdir(parents=True, exist_ok=False)
    with (args.output_directory / "request.json").open("x") as stream:
        json.dump(
            dict(
                arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                source_hashes=materials,
                policy_identity=identity,
                actual_source="existing saved PICO-derived robot-reference packets, not live headset",
                hardware_authorized=False,
                deployment_ready=False,
            ),
            stream,
            indent=2,
            sort_keys=True,
        )
    context = zmq.Context(io_threads=1)
    publisher = SavedPublisher(
        context,
        packets,
        scenario=args.scenario,
        fault_control=args.fault_control,
        pause_controls=args.pause_controls,
    )
    inbox = None
    try:
        publisher.start()
        inbox = LoopbackSubInbox(context, publisher.endpoint)
        arrays, report = run_stream(
            controller,
            inbox,
            source_controls=len(packets),
            tail_controls=args.tail_controls,
            first_control_index=packets[0]["control_source_frame_index"],
        )
        publisher.close()
        expected_fault = {"end-of-stream": "timeout", "pause": "timeout", "gap": "gap", "payload": "payload"}[
            args.scenario
        ]
        expected_tick = len(packets) if args.scenario == "end-of-stream" else args.fault_control
        delivery = [item["sha256"] for item in publisher.sent] == [item["sha256"] for item in inbox.receipts]
        scenario_passed = bool(
            report["physical_screen_passed"]
            and report["measured_compute_deadlines_passed"]
            and report["input_fault"] == expected_fault
            and report["input_fault_tick"] == expected_tick
            and delivery
            and publisher.error is None
        )
        report.update(
            scenario=args.scenario,
            expected_input_fault=expected_fault,
            expected_fault_tick=expected_tick,
            scenario_screen_passed=scenario_passed,
            publisher_error=publisher.error,
            all_sent_messages_received_in_order=delivery,
            publisher_origin_ns=publisher.origin_ns,
            endpoint=publisher.endpoint,
            policy_identity=identity,
            source_clip=str(args.packets.resolve()),
            source_clip_sha256=sha256_file(args.packets),
            source_motion_values_unchanged_except_explicit_fault_payload=True,
            source_timing_rebased_only=True,
            source_speed_changed=False,
            raw_headset_input_tested=False,
        )
        if args.baseline_trace is not None:
            with np.load(args.baseline_trace, allow_pickle=False) as baseline:
                report["baseline_trace_sha256"] = sha256_file(args.baseline_trace)
                report["baseline_exact_arrays"] = {
                    key: bool(np.array_equal(arrays[key], baseline[key]))
                    for key in ("qpos", "qvel", "simulation_time")
                }
        with (args.output_directory / "sent.json").open("x") as stream:
            json.dump(publisher.sent, stream, indent=2, allow_nan=False)
        report["sent_sha256"] = sha256_file(args.output_directory / "sent.json")
        save_result(args.output_directory, arrays, report, inbox)
        print(
            json.dumps(
                {key: value for key, value in report.items() if key not in ("policy_identity", "joint_names")},
                sort_keys=True,
            )
        )
        return 0 if scenario_passed else 1
    except Exception as error:
        with (args.output_directory / "failure.json").open("x") as stream:
            json.dump(
                dict(
                    error=f"{type(error).__name__}: {error}",
                    completed_controls=controller.completed,
                    publisher_error=publisher.error,
                    hardware_authorized=False,
                    deployment_ready=False,
                ),
                stream,
                indent=2,
            )
        raise
    finally:
        if publisher.thread.ident is not None:
            publisher.close()
        if inbox is not None:
            inbox.close()
        context.term()


if __name__ == "__main__":
    raise SystemExit(main())
