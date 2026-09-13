"""Full saved-packet equality and same-policy actual native physics preflight."""
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

from gear_sonic.scripts.record_g1_true23_saved_teleop_diagnostic import (
    build_recording_controller, preserve_calibrated_source_orientation, record,
)
from gear_sonic.scripts.replay_g1_true23_pico_packets_zmq import load_reference_packets
from gear_sonic.teleop.kinematic_reference import PreparedNative23Reference
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import sha256_file

ROOT = Path('/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
OUT = Path(__file__).parent / 'preflight_v1'
BASE = Path('/mnt/c/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1')
PAIR = ROOT / 'artifacts/g1_true23_frozen_lora/paired_encoder_20260905_v2/original_breadth25'
OLD = Path('/mnt/c/Users/camer/sonic23_sim_artifacts/public_vr_legs_20260910_v1/walk002_paired/measured_trace.npz')
args = SimpleNamespace(repository_root=ROOT.parent / 'GR00T-WholeBodyControl',
                       encoder_report=PAIR/'model_25.diagnostic.encoder.json',
                       decoder_report=PAIR/'model_25.diagnostic.decoder.json', legacy_unpaired_diagnostic=False)


def stats(values):
    a = np.asarray(values, dtype=np.float64) / 1e6
    return dict(count=len(a), median_ms=float(np.median(a)), p95_ms=float(np.percentile(a, 95)), max_ms=float(a.max()))


def timed_method(owner, name, rows, label):
    original = getattr(owner, name)

    def wrapped(*args, **kwargs):
        start = time.monotonic_ns()
        try:
            return original(*args, **kwargs)
        finally:
            rows.setdefault(label, []).append(time.monotonic_ns() - start)

    setattr(owner, name, wrapped)


def main():
    OUT.mkdir(exist_ok=False)
    controller, identity = build_recording_controller(args)
    prepared = PreparedNative23Reference(controller)
    equality = {}
    source = {}
    for name in ('walk002', 'walk003', 'walk008'):
        path = BASE / name / 'causal_packets.json'
        packets = load_reference_packets(path)
        source[name] = dict(path=str(path), sha256=sha256_file(path), packets=len(packets))
        elapsed = {'legacy': [], 'prepared': []}
        for packet in packets:
            start = time.monotonic_ns()
            expected = controller.retarget_pico_reference_packet(packet)
            elapsed['legacy'].append(time.monotonic_ns() - start)
            start = time.monotonic_ns()
            actual = prepared.retarget(packet)
            elapsed['prepared'].append(time.monotonic_ns() - start)
            assert expected == actual
        equality[name] = dict(all_packet_fields_exact=True, calls=len(packets), **{k:stats(v) for k,v in elapsed.items()})
        print(json.dumps({name: equality[name]}), flush=True)
    assert controller.completed == 0 and controller.data.time == 0
    phases = {}
    packets = load_reference_packets(BASE / 'walk002/causal_packets.json')
    for name in ('legacy', 'prepared'):
        c, _ = build_recording_controller(args)
        if name == 'prepared':
            fk = PreparedNative23Reference(c)
            fk.install(c)
        rows = {}
        for method in ('reference_root_height', 'retarget_pico_reference_packet', '_policy_frame', 'step'):
            timed_method(c, method, rows, method)
        timed_method(c.policy, 'infer', rows, 'policy_inference')
        preserve_calibrated_source_orientation(c, packets[0])
        trace, result = record(c, packets)
        with np.load(OLD, allow_pickle=False) as baseline:
            exact = {key:bool(np.array_equal(value, baseline[key])) for key,value in trace.items()}
        assert all(exact.values()), exact
        with (OUT / f'{name}.npz').open('xb') as stream:
            np.savez_compressed(stream, **trace, **{key+'_duration_ns':np.asarray(value) for key,value in rows.items()})
        phases[name] = dict(integration=result, baseline_exact=exact, stages={k:stats(v) for k,v in rows.items()},
                            trace_sha256=sha256_file(OUT/f'{name}.npz'))
        print(json.dumps({name:phases[name]}), flush=True)
    result = dict(kind='native23_prepared_reference_fk_preflight_v1', passed=True, packet_equality=equality,
                  sources=source, native_physics=phases, policy_identity=identity, baseline_sha256=sha256_file(OLD),
                  source_files={str(p):sha256_file(p) for p in (Path(__file__), Path(__file__).parent/'EXPERIMENT.md',
                              ROOT/'gear_sonic/teleop/kinematic_reference.py')},
                  actual_references_or_policy_changed=False, tracking_improved=False,
                  deployment_ready=False, hardware_authorized=False)
    with (OUT/'report.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
