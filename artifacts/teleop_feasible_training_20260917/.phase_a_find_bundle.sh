#!/usr/bin/env bash
set -euo pipefail
find /mnt/e/codex-artifacts /mnt/e/codex_sonic_runtime -type f -name native_prepared.xml -printf '%h\n' 2>/dev/null | sort -u \
  > /mnt/e/codex-artifacts/teleop_feasible_training_20260917/native_prepared_bundles.txt
