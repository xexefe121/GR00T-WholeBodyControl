#!/usr/bin/env bash
set -euo pipefail
source /root/venvs/teleop23/bin/activate
python /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_feasible_training_20260917/.phase_a_inventory.py \
  > /mnt/e/codex-artifacts/teleop_feasible_training_20260917/inventory_machine.json
