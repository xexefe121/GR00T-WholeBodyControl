"""No-spin network parity and full source/EOF-balance actual physics equality."""

import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

from gear_sonic.scripts.record_g1_true23_saved_teleop_diagnostic import (
    build_recording_controller,
    preserve_calibrated_source_orientation,
    record,
)
from gear_sonic.scripts.replay_g1_true23_pico_packets_zmq import load_reference_packets
from gear_sonic.teleop.cpu_paced_inference import prepare_nonspinning_sessions
from gear_sonic.teleop.kinematic_reference import PreparedNative23Reference
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import sha256_file

ROOT = Path("/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof")
OUT = Path(__file__).parent / "preflight_nospin_v1"
PAIR = ROOT / "artifacts/g1_true23_frozen_lora/paired_encoder_20260905_v2/original_breadth25"
SOURCE = Path("/mnt/c/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/walk002/causal_packets.json")
OLD = Path(__file__).parent / "actual_v1/end-of-stream-2/trace.npz"
TAPE = ROOT / "artifacts/g1_true23_vr_paced_20260910_v1/profile_measured_inputs.npz"


def main():
    OUT.mkdir(exist_ok=False)
    args = SimpleNamespace(
        repository_root=ROOT.parent / "GR00T-WholeBodyControl",
        encoder_report=PAIR / "model_25.diagnostic.encoder.json",
        decoder_report=PAIR / "model_25.diagnostic.decoder.json",
        legacy_unpaired_diagnostic=False,
    )
    c, identity = build_recording_controller(args)
    identity = prepare_nonspinning_sessions(c, identity, args.repository_root)
    PreparedNative23Reference(c).install(c)
    durations = []
    with np.load(TAPE, allow_pickle=False) as tape:
        for encoder, history, expected_raw, expected_decoder in zip(
            tape["encoder"], tape["history"], tape["raw"], tape["decoder"], strict=True
        ):
            start = time.monotonic_ns()
            raw, decoder = c.policy.infer(encoder, history)
            durations.append(time.monotonic_ns() - start)
            np.testing.assert_array_equal(raw, expected_raw)
            np.testing.assert_array_equal(decoder, expected_decoder)
    packets = load_reference_packets(SOURCE)
    preserve_calibrated_source_orientation(c, packets[0])
    trace, integration = record(c, packets)
    assert integration["passed"] and c.completed == 656
    q, v, times = list(trace["qpos"]), list(trace["qvel"]), list(trace["simulation_time"])
    c.activate_fallback("transport_timeout")
    for _ in range(250):
        c.step(np.zeros(267, dtype=np.float32))
        q.append(c.data.qpos.copy())
        v.append(c.data.qvel.copy())
        times.append(float(c.data.time))
    actual = dict(qpos=np.asarray(q), qvel=np.asarray(v), simulation_time=np.asarray(times))
    with np.load(OLD, allow_pickle=False) as baseline:
        exact = {key: bool(np.array_equal(value, baseline[key])) for key, value in actual.items()}
    assert all(exact.values()), exact
    with (OUT / "trace.npz").open("xb") as stream:
        np.savez_compressed(stream, **actual, inference_duration_ns=np.asarray(durations))
    result = dict(
        kind="native23_nonspinning_full_equality_preflight_v1",
        passed=True,
        network_samples_exact=len(durations),
        source_controls=656,
        balance_controls=250,
        full_native_physics_control_states_exact=exact,
        actual_physics_substeps=9060,
        identity=identity,
        inference_was_warmed_by_parity_check_in_this_virtual_test_only=True,
        no_runtime_warmup_claim=True,
        deployment_ready=False,
        tracking_improved=False,
        hardware_authorized=False,
        trace_sha256=sha256_file(OUT / "trace.npz"),
        sources={
            str(p): sha256_file(p)
            for p in (
                Path(__file__),
                SOURCE,
                OLD,
                TAPE,
                Path(__file__).parent / "NONSPIN_PLAN.md",
                ROOT / "gear_sonic/teleop/cpu_paced_inference.py",
            )
        },
    )
    with (OUT / "report.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(
        json.dumps({key: value for key, value in result.items() if key not in ("sources", "identity")}), flush=True
    )


if __name__ == "__main__":
    main()
