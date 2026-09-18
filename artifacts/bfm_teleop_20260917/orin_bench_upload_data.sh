#!/usr/bin/env bash
set -euo pipefail
repo=/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof
helper="$repo/gear_sonic/scripts/fix5_onboard_ssh.sh"
put() { bash "$helper" run "cat > /home/unitree/bfm_orin_bench/$2" < "$1"; }
put "$repo/artifacts/g1_true23_six_hour_replan_20260910_v1/bfmzero_inference_v1/inference.safetensors" weights/inference.safetensors
put "$repo/artifacts/g1_true23_six_hour_replan_20260910_v1/bfmzero_inspect_v1/config.yaml" weights/config.yaml
put "$repo/artifacts/bfm_teleop_20260917/orin_local_input/data/pico_teleop_50hz.npz" data/pico_teleop_50hz.npz
put "$repo/artifacts/bfm_teleop_20260917/orin_local_input/data/windows_c1_reference.npz" data/windows_c1_reference.npz
put /mnt/e/codex-artifacts/bfm_teleop_20260917/fix4/pico_lowstate_500hz.flatbin data/pico_lowstate_500hz.flatbin
