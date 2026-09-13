#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
base=/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/direct_target_width512_expert_recovery_v2
printf '{"linux_pid":%s,"process_identity":"bash exec replaced by selected Python"}\n' "$$" > "$base/recovery_process_v1/linux_process.json"
exec /mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python -B -u "$base/source_snapshot_v1/run_width251_actual_oracle.py"
