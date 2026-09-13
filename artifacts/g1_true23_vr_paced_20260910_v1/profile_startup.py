"""No dynamics: time startup work before changing stream admission."""
import json
from pathlib import Path
from types import SimpleNamespace
import time

from gear_sonic.scripts.record_g1_true23_saved_teleop_diagnostic import build_recording_controller
from gear_sonic.scripts.replay_g1_true23_pico_packets_zmq import load_reference_packets, rebase_reference_packet_time
from gear_sonic.utils.g1_true23_frozen_lora_live_teleop import validate_live_packet
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256

ROOT = Path('/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
PAIR = ROOT / 'artifacts/g1_true23_frozen_lora/paired_encoder_20260905_v2/original_breadth25'
args = SimpleNamespace(repository_root=Path('/mnt/z/codex/GR00T-WholeBodyControl'),
                       encoder_report=PAIR/'model_25.diagnostic.encoder.json',
                       decoder_report=PAIR/'model_25.diagnostic.decoder.json', legacy_unpaired_diagnostic=False)
controller, _ = build_recording_controller(args)
packet = load_reference_packets(Path('/mnt/c/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/walk002/causal_packets.json'))[0]
result = []
for i in range(3):
    before = time.monotonic_ns()
    digest = compiled_model_sha256(controller.model)
    after = time.monotonic_ns()
    rebased = rebase_reference_packet_time(packet, first_control_index=packet['control_source_frame_index'],
                                          first_control_monotonic_ns=time.monotonic_ns())
    valid_before = time.monotonic_ns()
    validate_live_packet(rebased, previous=None, maximum_age_ns=100_000_000)
    result.append(dict(repetition=i, compiled_model_hash_ms=(after-before)/1e6,
                       validation_ms=(time.monotonic_ns()-valid_before)/1e6, compiled_model_sha256=digest))
report = dict(records=result, controller_completed=controller.completed, simulation_time=controller.data.time,
              no_physics_steps=True, no_safeguards_changed=True)
with (Path(__file__).parent/'startup_profile.json').open('x') as stream:
    json.dump(report, stream, indent=2)
print(json.dumps(report))
