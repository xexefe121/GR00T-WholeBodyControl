#!/usr/bin/env bash
set -euo pipefail
source /root/venvs/teleop23/bin/activate
cd /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof
python gear_sonic/scripts/audit_g1_true23_feasible_reference_motion.py \
  --asset-root /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof \
  --motion walk001=/mnt/e/codex-artifacts/teleop_feasible_training_20260917/twist2_import/walk001/motion.native23.npz \
  --motion walk002=/mnt/e/codex-artifacts/teleop_feasible_training_20260917/twist2_import/walk002/motion.native23.npz \
  --motion walk003=/mnt/e/codex-artifacts/teleop_feasible_training_20260917/twist2_import/walk003/motion.native23.npz \
  --motion walk004=/mnt/e/codex-artifacts/teleop_feasible_training_20260917/twist2_import/walk004/motion.native23.npz \
  --motion walk005=/mnt/e/codex-artifacts/teleop_feasible_training_20260917/twist2_import/walk005/motion.native23.npz \
  --motion walk006=/mnt/e/codex-artifacts/teleop_feasible_training_20260917/twist2_import/walk006/motion.native23.npz \
  --motion walk007=/mnt/e/codex-artifacts/teleop_feasible_training_20260917/twist2_import/walk007/motion.native23.npz \
  --motion walk008=/mnt/e/codex-artifacts/teleop_feasible_training_20260917/twist2_import/walk008/motion.native23.npz \
  --motion walk009=/mnt/e/codex-artifacts/teleop_feasible_training_20260917/twist2_import/walk009/motion.native23.npz \
  --motion walk010=/mnt/e/codex-artifacts/teleop_feasible_training_20260917/twist2_import/walk010/motion.native23.npz \
  --motion pico=/mnt/e/codex-artifacts/sonic23_teleop_six_hour_20260910/mjbatch_intent_inputs_v1/pico/reference.npz \
  --output /mnt/e/codex-artifacts/teleop_feasible_training_20260917/feasibility_audit.json
