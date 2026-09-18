#!/usr/bin/env bash
set -euo pipefail
find /mnt/e/codex-artifacts -maxdepth 3 -type d \( -iname '*pico*' -o -iname '*twist*' -o -iname '*motion*' \) -print | sort \
  > /mnt/e/codex-artifacts/teleop_feasible_training_20260917/e_relevant_directories.txt
