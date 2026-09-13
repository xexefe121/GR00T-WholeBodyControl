#!/usr/bin/env bash
set -euo pipefail

cd /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof
source /root/.venvs/g1_true23_mjlab/bin/activate

root=/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof
asset_root=/mnt/z/codex/GR00T-WholeBodyControl
run="$root/artifacts/g1_true23_frozen_lora/original_sonic_happy_specialist_low_lr_seed20260818_v1"
eval_dir="$run/eval"
suite="$root/artifacts/g1_true23_frozen_lora/original_sonic_parity_happy_v1/happy_focus_suite.json"
mkdir -p "$eval_dir"

for update in 10 20 30 40 50; do
  stem="$eval_dir/model_${update}.diagnostic"
  python -m gear_sonic.scripts.materialize_g1_true23_frozen_lora_diagnostic \
    --checkpoint "$run/checkpoints/frozen_lora_model_${update}.pt" \
    --warm-start "$asset_root/sonic_release/g1_23dof_rev_1_0_low_latency_init.pt" \
    --source-checkpoint "$asset_root/low_latency/last.pt" \
    --output "$stem.pt"
  python -m gear_sonic.scripts.export_g1_true23_frozen_lora_diagnostic_decoder \
    --diagnostic-policy "$stem.pt" \
    --output "$stem.decoder.onnx" \
    --report "$stem.decoder.json"
  python -m gear_sonic.scripts.evaluate_g1_true23_frozen_lora_suite \
    --repository-root "$asset_root" \
    --decoder-report "$stem.decoder.json" \
    --suite "$suite" \
    --output-dir "$eval_dir/model_${update}_happy_focus"
done
