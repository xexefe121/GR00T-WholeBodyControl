#!/usr/bin/env bash
# Run a full-body native23 teleop session in simulation from a recorded PICO
# source, across the localhost ZMQ transport, into MuJoCo.
#
# Simulator only. Opens no DDS channel, no Unitree robot channel, and publishes
# no robot command. A PICO headset is not required and is not used.
#
# Usage, from Windows:
#   wsl.exe -d Ubuntu-22.04 -- bash /root/run_teleop_sim.sh [output-directory]
#
# Default output directory is timestamped under E:\codex-artifacts\.

set -uo pipefail

ROOT=/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof
VENV=/root/venvs/teleop23
DEST="${1:-/mnt/e/codex-artifacts/teleop_sim_$(date +%Y%m%d_%H%M%S)}"

PACKETS="$ROOT/artifacts/g1_true23_frozen_lora/physical_dance_v1/original_sonic_happy.true23.causal_packets.json"
ENC="$ROOT/artifacts/g1_true23_frozen_lora/paired_encoder_20260905_v2/original_breadth25/model_25.diagnostic.encoder.json"
DEC="$ROOT/artifacts/g1_true23_frozen_lora/paired_encoder_20260905_v2/original_breadth25/model_25.diagnostic.decoder.json"

# Publisher warm-up must cover the consumer's model load; the consumer's receive
# timeout must then exceed whatever warm-up remains. Raising warm-up alone makes
# the consumer time out before the first packet is ever sent.
WARMUP_S=15
RECEIVE_TIMEOUT_MS=25000
STARTUP_TIMEOUT_MS=30000
STEPS=535

for required in "$VENV/bin/activate" "$PACKETS" "$ENC" "$DEC"; do
    if [ ! -e "$required" ]; then
        echo "missing prerequisite: $required" >&2
        exit 1
    fi
done

cd "$ROOT"
# shellcheck disable=SC1090
source "$VENV/bin/activate"
mkdir -p "$DEST"

echo "output directory: $DEST"

python gear_sonic/scripts/replay_g1_true23_pico_packets_zmq.py \
    --packets "$PACKETS" \
    --bind tcp://127.0.0.1:5557 \
    --subscriber-warmup-s "$WARMUP_S" \
    --output "$DEST/publisher.json" \
    > "$DEST/publisher.log" 2>&1 &
PUB_PID=$!

python gear_sonic/scripts/run_g1_true23_frozen_lora_live_teleop.py \
    --repository-root "$ROOT" \
    --encoder-report "$ENC" \
    --decoder-report "$DEC" \
    --endpoint tcp://127.0.0.1:5557 \
    --steps "$STEPS" \
    --startup-timeout-ms "$STARTUP_TIMEOUT_MS" \
    --receive-timeout-ms "$RECEIVE_TIMEOUT_MS" \
    --output "$DEST/consumer.json" \
    --trace-output "$DEST/measured_trace.npz" \
    > "$DEST/consumer.log" 2>&1
CONSUMER_STATUS=$?

wait "$PUB_PID" 2>/dev/null

python - "$DEST/consumer.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    report = json.load(stream)

fields = (
    "passed",
    "completed_live_transitions",
    "attempted_live_transitions",
    "observed_transport_fault",
    "fallback_transition_count",
    "minimum_base_height_m",
    "maximum_base_tilt_rad",
    "maximum_reference_age_ns",
    "maximum_reference_age_gate_ns",
    "live_transport_proven",
    "live_headset_source_proven",
    "tracking_fidelity_qualified",
    "deployment_ready",
)
for field in fields:
    if field in report:
        print(f"{field:34} {report[field]}")
print(f"{'authorization':34} {report.get('authorization')}")
PY

exit "$CONSUMER_STATUS"
