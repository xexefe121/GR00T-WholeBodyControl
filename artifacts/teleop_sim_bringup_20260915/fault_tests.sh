#!/usr/bin/env bash
# Transport-fault behaviour for the frozen-LoRA live teleop consumer.
# Simulator only: no DDS, no robot channel, localhost transport.
set -uo pipefail

ROOT=/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof
DEST=/mnt/e/codex-artifacts/teleop_sim_bringup_20260915
PACKETS="$ROOT/artifacts/g1_true23_frozen_lora/physical_dance_v1/original_sonic_happy.true23.causal_packets.json"
ENC="$ROOT/artifacts/g1_true23_frozen_lora/paired_encoder_20260905_v2/original_breadth25/model_25.diagnostic.encoder.json"
DEC="$ROOT/artifacts/g1_true23_frozen_lora/paired_encoder_20260905_v2/original_breadth25/model_25.diagnostic.decoder.json"

cd "$ROOT"
source /root/venvs/teleop23/bin/activate
mkdir -p "$DEST"

for FAULT in timeout gap stale; do
    echo "=== fault: $FAULT ==="
    rm -f "$DEST/fault_${FAULT}_pub.json" "$DEST/fault_${FAULT}_sub.json"

    python gear_sonic/scripts/replay_g1_true23_pico_packets_zmq.py \
        --packets "$PACKETS" \
        --bind tcp://127.0.0.1:5557 \
        --subscriber-warmup-s 15 \
        --fault "$FAULT" \
        --fault-offset 120 \
        --output "$DEST/fault_${FAULT}_pub.json" \
        > "$DEST/fault_${FAULT}_pub.log" 2>&1 &
    PUB_PID=$!

    python gear_sonic/scripts/run_g1_true23_frozen_lora_live_teleop.py \
        --repository-root "$ROOT" \
        --encoder-report "$ENC" \
        --decoder-report "$DEC" \
        --endpoint tcp://127.0.0.1:5557 \
        --steps 535 \
        --startup-timeout-ms 30000 \
        --receive-timeout-ms 25000 \
        --expected-transport-fault "$FAULT" \
        --output "$DEST/fault_${FAULT}_sub.json" \
        > "$DEST/fault_${FAULT}_sub.log" 2>&1
    echo "  consumer exit: $?"

    wait "$PUB_PID" 2>/dev/null
    echo "  publisher exit: $?"
done

echo "=== done ==="
