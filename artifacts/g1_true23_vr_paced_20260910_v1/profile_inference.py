"""Compare CPU scheduling options on identical captured inputs; no policy fit."""
import gc
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import onnxruntime as ort

from gear_sonic.scripts.record_g1_true23_saved_teleop_diagnostic import (
    build_recording_controller, preserve_calibrated_source_orientation, record,
)
from gear_sonic.scripts.replay_g1_true23_pico_packets_zmq import load_reference_packets
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import sha256_file
from gear_sonic.utils.g1_true23_sonic_library_replay import ExactHashSonicPolicy

ROOT = Path('/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
OUT = Path(__file__).parent
PAIR = ROOT / 'artifacts/g1_true23_frozen_lora/paired_encoder_20260905_v2/original_breadth25'
args = SimpleNamespace(repository_root=Path('/mnt/z/codex/GR00T-WholeBodyControl'),
                       encoder_report=PAIR/'model_25.diagnostic.encoder.json',
                       decoder_report=PAIR/'model_25.diagnostic.decoder.json', legacy_unpaired_diagnostic=False)
controller, identity = build_recording_controller(args)
source = Path('/mnt/c/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/walk002/causal_packets.json')
packets = load_reference_packets(source)
policy = controller.policy
samples, events = [], []
def gc_event(phase, info):
    events.append(dict(time_ns=time.monotonic_ns(), phase=phase, generation=info['generation']))
gc.callbacks.append(gc_event)
class Capture:
    def infer(self, encoder, history):
        before = time.monotonic_ns()
        raw, decoder = policy.infer(encoder, history)
        after = time.monotonic_ns()
        samples.append((encoder.copy(), history.copy(), raw.copy(), decoder.copy(), before, after))
        return raw, decoder
controller.policy = Capture()
preserve_calibrated_source_orientation(controller, packets[0])
traces, result = record(controller, packets)
gc.callbacks.remove(gc_event)
baseline = Path('/mnt/c/Users/camer/sonic23_sim_artifacts/public_vr_legs_20260910_v1/walk002_paired/measured_trace.npz')
with np.load(baseline, allow_pickle=False) as old:
    exact = {k:bool(np.array_equal(v, old[k])) for k,v in traces.items()}
assert all(exact.values()), exact
with (OUT/'profile_measured_inputs.npz').open('xb') as stream:
    np.savez_compressed(stream, encoder=np.stack([s[0] for s in samples]), history=np.stack([s[1] for s in samples]),
                       raw=np.stack([s[2] for s in samples]), decoder=np.stack([s[3] for s in samples]),
                       before_ns=np.asarray([s[4] for s in samples]), after_ns=np.asarray([s[5] for s in samples]))
profiles = {}
for name in ('default', 'no_spin', 'single_thread_no_spin'):
    options = None
    if name != 'default':
        options = ort.SessionOptions()
        options.add_session_config_entry('session.intra_op.allow_spinning', '0')
        options.add_session_config_entry('session.inter_op.allow_spinning', '0')
        if name == 'single_thread_no_spin':
            options.intra_op_num_threads = options.inter_op_num_threads = 1
    test = ExactHashSonicPolicy(encoder_path=Path(identity['diagnostic_pair']['encoder']['path']),
                               decoder_path=Path(identity['diagnostic_pair']['decoder']['path']),
                               expected_encoder_sha256=identity['encoder_sha256'],
                               expected_decoder_sha256=identity['decoder_sha256'], session_options=options)
    rows = []
    for sample in samples:
        before = time.monotonic_ns()
        raw, decoder = test.infer(sample[0], sample[1])
        duration = time.monotonic_ns()-before
        rows.append(dict(duration_ns=duration, raw_exact=bool(np.array_equal(raw, sample[2])),
                         decoder_exact=bool(np.array_equal(decoder, sample[3])),
                         raw_max_abs=float(np.max(np.abs(raw-sample[2])))))
    durations = [r['duration_ns']/1e6 for r in rows]
    profiles[name] = dict(rows=rows, maximum_ms=max(durations), p95_ms=float(np.percentile(durations,95)),
                          all_raw_exact=all(r['raw_exact'] for r in rows),
                          all_decoder_exact=all(r['decoder_exact'] for r in rows),
                          maximum_raw_abs=max(r['raw_max_abs'] for r in rows))
    del test
    gc.collect()
result = dict(captured_full_walk=result, baseline_arrays_exact=exact,
              source_sha256=sha256_file(source), original_baseline_sha256=sha256_file(baseline),
              capture_sha256=sha256_file(OUT/'profile_measured_inputs.npz'), profiles=profiles,
              gc_events_during_capture=events, policy_identity=identity,
              profiling_not_real_time_qualification=True, tracking_improved=False,
              hardware_authorized=False, deployment_ready=False)
with (OUT/'inference_profile.json').open('x') as stream:
    json.dump(result, stream, indent=2, allow_nan=False)
print(json.dumps({k:{n:v for n,v in p.items() if n!='rows'} for k,p in profiles.items()}))
