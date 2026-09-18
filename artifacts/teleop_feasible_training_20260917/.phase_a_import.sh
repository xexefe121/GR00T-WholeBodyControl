#!/usr/bin/env bash
set -euo pipefail

source /root/venvs/teleop23/bin/activate
cd /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof
mkdir -p /mnt/e/codex-artifacts/teleop_feasible_training_20260917/twist2_import
for stem in 001 002 003 004 005 006 007 008 009 010; do
  python gear_sonic/scripts/prepare_g1_true23_twist2_replay.py \
    --source "/mnt/z/codex/twist2_inspect/assets/example_motions/0807_yanjie_walk_${stem}.pkl" \
    --asset-root /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof \
    --output-directory "/mnt/e/codex-artifacts/teleop_feasible_training_20260917/twist2_import/walk${stem}"
done
