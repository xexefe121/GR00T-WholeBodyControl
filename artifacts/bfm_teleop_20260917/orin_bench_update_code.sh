#!/usr/bin/env bash
set -euo pipefail
repo=/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof
helper="$repo/gear_sonic/scripts/fix5_onboard_ssh.sh"
bash "$helper" run 'rm -f /home/unitree/bfm_orin_bench/reports/c1.json /home/unitree/bfm_orin_bench/reports/benchmark.log'
bash "$helper" run 'cat > /home/unitree/bfm_orin_bench/src/orin_bench.py' < "$repo/artifacts/bfm_teleop_20260917/orin_bench.py"
bash "$helper" run 'nohup /home/unitree/bfm_orin_bench/run_benchmark.sh > /home/unitree/bfm_orin_bench/reports/benchmark.log 2>&1 < /dev/null &'
