"""Fixed six-case actual-clock evaluation; retain failures, never choose a rerun."""

import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import zmq

from gear_sonic.scripts.record_g1_true23_saved_teleop_diagnostic import build_recording_controller
from gear_sonic.scripts.replay_g1_true23_pico_packets_zmq import load_reference_packets
from gear_sonic.scripts.run_g1_true23_paced_sim import prepare_stream, run_stream, save_result
from gear_sonic.scripts.test_g1_true23_paced_saved_stream import SavedPublisher
from gear_sonic.teleop.kinematic_reference import PreparedNative23Reference
from gear_sonic.teleop.cpu_paced_inference import prepare_nonspinning_sessions
from gear_sonic.teleop.paced_sim_runtime import LoopbackSubInbox
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import sha256_file
from gear_sonic.utils.g1_true23_generalist_source_closure import collect_local_source_closure

ROOT = Path("/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof")
OUT = Path(__file__).parent / "actual_nospin_v1"
SOURCE = Path("/mnt/c/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/walk002/causal_packets.json")
OLD = Path("/mnt/c/Users/camer/sonic23_sim_artifacts/public_vr_legs_20260910_v1/walk002_paired/measured_trace.npz")
PAIR = ROOT / "artifacts/g1_true23_frozen_lora/paired_encoder_20260905_v2/original_breadth25"
CASES = ("end-of-stream-1", "pause", "gap", "payload", "end-of-stream-2", "end-of-stream-3")


def instrument(owner, method, name, controller, rows):
    fn = getattr(owner, method)

    def wrapped(*args, **kwargs):
        index, start = controller.completed, time.monotonic_ns()
        try:
            return fn(*args, **kwargs)
        finally:
            rows.append((name, index, time.monotonic_ns() - start))

    setattr(owner, method, wrapped)


def main():
    preflight_path = Path(__file__).parent / "preflight_v1/report.json"
    preflight = json.loads(preflight_path.read_text())
    assert preflight["passed"]
    parity_path = Path(__file__).parent / "preflight_nospin_v1/report.json"
    parity = json.loads(parity_path.read_text())
    assert parity["passed"]
    for path, expected in parity["sources"].items():
        assert sha256_file(Path(path)) == expected
    for path, expected in preflight["source_files"].items():
        assert sha256_file(Path(path)) == expected
    packets = load_reference_packets(SOURCE)
    with np.load(OLD, allow_pickle=False) as archive:
        baseline = {key: archive[key].copy() for key in ("qpos", "qvel", "simulation_time")}
    materials = {
        str(path): sha256_file(path)
        for path in collect_local_source_closure(
            ROOT,
            [
                ROOT / "gear_sonic/scripts/test_g1_true23_paced_saved_stream.py",
                ROOT / "gear_sonic/teleop/kinematic_reference.py",
                ROOT / "gear_sonic/teleop/cpu_paced_inference.py",
            ],
        )
        .as_source_files(ROOT)
        .values()
    }
    materials.update(
        {
            str(path): sha256_file(path)
            for path in (
                SOURCE,
                OLD,
                preflight_path,
                parity_path,
                Path(__file__),
                Path(__file__).parent / "EXPERIMENT.md",
                Path(__file__).parent / "NONSPIN_PLAN.md",
            )
        }
    )
    OUT.mkdir(exist_ok=False)
    summary = []
    for case in CASES:
        directory = OUT / case
        directory.mkdir()
        args = SimpleNamespace(
            repository_root=ROOT.parent / "GR00T-WholeBodyControl",
            encoder_report=PAIR / "model_25.diagnostic.encoder.json",
            decoder_report=PAIR / "model_25.diagnostic.decoder.json",
            legacy_unpaired_diagnostic=False,
        )
        c, identity = build_recording_controller(args)
        identity = prepare_nonspinning_sessions(c, identity, args.repository_root)
        PreparedNative23Reference(c).install(c)
        prepared = prepare_stream(c)
        stages = []
        instrument(c, "retarget_pico_reference_packet", "retarget", c, stages)
        instrument(c.policy, "infer", "sonic_inference", c, stages)
        instrument(c, "step", "controller_step", c, stages)
        scenario = "end-of-stream" if case.startswith("end-of-stream") else case
        with (directory / "request.json").open("x") as stream:
            json.dump(
                dict(
                    case=case,
                    scenario=scenario,
                    all_planned_cases=CASES,
                    source_hashes=materials,
                    policy_identity=identity,
                    inference_warmup_added=False,
                    hardware_authorized=False,
                ),
                stream,
                indent=2,
                allow_nan=False,
            )
        context = zmq.Context(io_threads=1)
        publisher = SavedPublisher(context, packets, scenario=scenario, fault_control=200, pause_controls=25)
        inbox = None
        try:
            publisher.start()
            inbox = LoopbackSubInbox(context, publisher.endpoint)
            arrays, report = run_stream(
                prepared,
                inbox,
                source_controls=len(packets),
                tail_controls=250,
                first_control_index=packets[0]["control_source_frame_index"],
            )
            publisher.close()
            count = report["completed_sonic_controls"]
            exact = {
                key: bool(np.array_equal(arrays[key][: count + 1], baseline[key][: count + 1])) for key in baseline
            }
            expected_tick = len(packets) if scenario == "end-of-stream" else 200
            expected_fault = {"end-of-stream": "timeout", "pause": "timeout", "gap": "gap", "payload": "payload"}[
                scenario
            ]
            delivery = [x["sha256"] for x in publisher.sent] == [x["sha256"] for x in inbox.receipts]
            report.update(
                case=case,
                policy_identity=identity,
                source_fields_preserved_by_preflight=True,
                expected_input_fault_tick=expected_tick,
                expected_input_fault=expected_fault,
                sonic_prefix_baseline_exact=exact,
                all_sent_messages_received_in_order=delivery,
                publisher_error=publisher.error,
                scenario_passed=bool(
                    report["physical_screen_passed"]
                    and report["measured_compute_deadlines_passed"]
                    and count == expected_tick
                    and report["input_fault_tick"] == expected_tick
                    and report["input_fault"] == expected_fault
                    and delivery
                    and publisher.error is None
                    and all(exact.values())
                ),
            )
            for label in ("retarget", "sonic_inference", "controller_step"):
                rows = np.asarray([(i, d) for n, i, d in stages if n == label], dtype=np.int64).reshape(-1, 2)
                arrays[label + "_index"] = rows[:, 0]
                arrays[label + "_duration_ns"] = rows[:, 1]
                report[label + "_timing"] = dict(
                    calls=len(rows),
                    max_ms=float(rows[:, 1].max(initial=0)) / 1e6,
                    p95_ms=float(np.percentile(rows[:, 1], 95)) / 1e6 if len(rows) else None,
                    slowest_control=int(rows[np.argmax(rows[:, 1]), 0]) if len(rows) else None,
                )
            save_result(directory, arrays, report, inbox)
            with (directory / "publisher.json").open("x") as stream:
                json.dump(dict(sent=publisher.sent, error=publisher.error), stream, indent=2, allow_nan=False)
            row = {
                key: report[key]
                for key in (
                    "case",
                    "scenario_passed",
                    "completed_sonic_controls",
                    "fallback_controls",
                    "runtime_fault",
                    "runtime_fault_tick",
                    "input_fault",
                    "input_fault_tick",
                    "maximum_execution_ms",
                    "execution_p95_ms",
                    "retarget_timing",
                    "sonic_inference_timing",
                    "physical_screen_passed",
                )
            }
            row["report_sha256"] = sha256_file(directory / "report.json")
            summary.append(row)
            print(json.dumps(row), flush=True)
        except Exception as error:
            with (directory / "failure.json").open("x") as stream:
                json.dump(
                    dict(error=f"{type(error).__name__}: {error}", completed_controls=c.completed),
                    stream,
                    indent=2,
                )
            raise
        finally:
            publisher.close()
            if inbox is not None:
                inbox.close()
            context.term()
    result = dict(
        kind="native23_nonspinning_fixed_six_case_pacing_v1",
        cases=summary,
        all_scenarios_passed=all(row["scenario_passed"] for row in summary),
        tracking_improved=False,
        tracking_qualified=False,
        deployment_ready=False,
        hardware_authorized=False,
    )
    with (OUT / "summary.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
