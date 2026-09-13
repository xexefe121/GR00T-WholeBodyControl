#!/usr/bin/env bash
set -euo pipefail

cd /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof
source /root/.venvs/g1_true23_mjlab/bin/activate

run_root=/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/g1_true23_frozen_lora
run_dir="$run_root/original_sonic_happy_rehearsal_low_lr_seed20260818_v1"
log_file="$run_root/original_sonic_happy_rehearsal_low_lr_seed20260818_v1.train.log"
exit_file="$run_root/original_sonic_happy_rehearsal_low_lr_seed20260818_v1.exit"

set +e
python -m gear_sonic.scripts.train_g1_23dof_mjlab_frozen_lora \
  --phase polish \
  --behavior-bank "$run_root/original_sonic_rehearsal_corpus_v1/behavior_bank.json" \
  --adapter-init "$run_root/pico_internet_breadth_100_seed20260815_canonical_v2/checkpoints/frozen_lora_model_25.pt" \
  --source-checkpoint /mnt/z/codex/GR00T-WholeBodyControl/low_latency/last.pt \
  --spans "$run_root/original_sonic_rehearsal_corpus_v1/corpus.spans.json" \
  train \
  --warm-start /mnt/z/codex/GR00T-WholeBodyControl/sonic_release/g1_23dof_rev_1_0_low_latency_init.pt \
  --motion-file "$run_root/original_sonic_rehearsal_corpus_v1/corpus.npz" \
  --motion-metadata "$run_root/original_sonic_rehearsal_corpus_v1/corpus.recovery.json" \
  --run-dir "$run_dir" \
  --num-envs 64 \
  --iterations 50 \
  --save-interval 10 \
  --learning-rate 5e-7 \
  --seed 20260818 \
  >"$log_file" 2>&1
status=$?
set -e
printf '%s\n' "$status" >"$exit_file"
exit "$status"
