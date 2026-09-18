#!/usr/bin/env bash
set -euo pipefail
repo=/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof
helper="$repo/gear_sonic/scripts/fix5_onboard_ssh.sh"
bash "$helper" run "cat > /home/unitree/bfm_orin_bench/run_benchmark.sh" < "$repo/artifacts/bfm_teleop_20260917/orin_bench_run.sh"
bash "$helper" run 'chmod 700 /home/unitree/bfm_orin_bench/run_benchmark.sh; nohup /home/unitree/bfm_orin_bench/run_benchmark.sh > /home/unitree/bfm_orin_bench/reports/benchmark.log 2>&1 < /dev/null &'
