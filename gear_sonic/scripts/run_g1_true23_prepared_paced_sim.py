"""Opt-in prepared/non-spinning localhost SIM receiver; NOT deployment ready.

Recorded-source completion is required for exit0, in addition to the unchanged
physical and measured-deadline screens. Input loss still latches balance but
cannot masquerade as completing the requested source. There is no robot/DDS,
headset service startup, physical mode handback or implicit SONIC re-entry.
"""

import json
from pathlib import Path

import zmq

from gear_sonic.scripts.record_g1_true23_saved_teleop_diagnostic import build_recording_controller
from gear_sonic.scripts.run_g1_true23_paced_sim import parse_args, prepare_stream, run_stream, save_result
from gear_sonic.teleop.cpu_paced_inference import prepare_nonspinning_sessions
from gear_sonic.teleop.kinematic_reference import PreparedNative23Reference
from gear_sonic.teleop.paced_sim_runtime import LoopbackSubInbox
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import sha256_file
from gear_sonic.utils.g1_true23_generalist_source_closure import collect_local_source_closure


def full_recorded_sim_success(report):
    return all(
        report.get(name) is True
        for name in ("physical_screen_passed", "measured_compute_deadlines_passed", "full_source_sonic_completed")
    )


def main(argv=None):
    args = parse_args(argv)
    args.legacy_unpaired_diagnostic = False
    args.output_directory.mkdir(parents=True, exist_ok=False)
    controller, identity = build_recording_controller(args)
    identity = prepare_nonspinning_sessions(controller, identity, args.repository_root)
    PreparedNative23Reference(controller).install(controller)
    prepared = prepare_stream(controller)
    root = Path(__file__).resolve().parents[2]
    sources = {
        str(path): sha256_file(path)
        for path in collect_local_source_closure(root, [Path(__file__)]).as_source_files(root).values()
    }
    with (args.output_directory / "request.json").open("x") as stream:
        json.dump(
            dict(
                arguments={
                    key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                },
                policy_identity=identity,
                source_hashes=sources,
                deployment_ready=False,
                hardware_authorized=False,
            ),
            stream,
            indent=2,
            allow_nan=False,
        )
    context = zmq.Context(io_threads=1)
    inbox = None
    try:
        inbox = LoopbackSubInbox(context, args.endpoint)
        arrays, report = run_stream(
            prepared,
            inbox,
            source_controls=args.source_controls,
            tail_controls=args.tail_controls,
            first_control_index=args.first_control_index,
        )
        report.update(
            policy_identity=identity,
            endpoint=args.endpoint,
            reference_fk="prepared_kinematics_only_exact_legacy_geometry",
            exit_zero_requires_full_source_and_timing_and_limited_physical_screen=True,
        )
        save_result(args.output_directory, arrays, report, inbox)
        print(json.dumps(report, sort_keys=True, allow_nan=False))
        return 0 if full_recorded_sim_success(report) else 1
    except Exception as error:
        with (args.output_directory / "failure.json").open("x") as stream:
            json.dump(
                dict(
                    error=f"{type(error).__name__}: {error}",
                    completed_controls=controller.completed,
                    deployment_ready=False,
                    hardware_authorized=False,
                ),
                stream,
                indent=2,
            )
        raise
    finally:
        if inbox is not None:
            inbox.close()
        context.term()


if __name__ == "__main__":
    raise SystemExit(main())
