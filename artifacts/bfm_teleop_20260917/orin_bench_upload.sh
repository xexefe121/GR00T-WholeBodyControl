#!/usr/bin/env bash
set -euo pipefail
repo=/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof
helper="$repo/gear_sonic/scripts/fix5_onboard_ssh.sh"
put() { bash "$helper" run "cat > /home/unitree/bfm_orin_bench/$2" < "$1"; }
bash "$helper" run 'mkdir -p /home/unitree/bfm_orin_bench/src/gear_sonic/utils /home/unitree/bfm_orin_bench/data/meshes /home/unitree/bfm_orin_bench/weights'
printf '' | bash "$helper" run 'cat > /home/unitree/bfm_orin_bench/src/gear_sonic/__init__.py'
printf '' | bash "$helper" run 'cat > /home/unitree/bfm_orin_bench/src/gear_sonic/utils/__init__.py'
put "$repo/artifacts/bfm_teleop_20260917/orin_bench_step1b.py" src/gear_sonic/utils/g1_true23_step1b_mujoco.py
put "$repo/gear_sonic/utils/g1_23dof_contract.py" src/gear_sonic/utils/g1_23dof_contract.py
put "$repo/gear_sonic/utils/g1_true23_bfmzero_inference.py" src/gear_sonic/utils/g1_true23_bfmzero_inference.py
put "$repo/gear_sonic/utils/g1_true23_bfm_imu_odometry.py" src/gear_sonic/utils/g1_true23_bfm_imu_odometry.py
put "$repo/gear_sonic/utils/g1_true23_bfmzero_stream.py" src/gear_sonic/utils/g1_true23_bfmzero_stream.py
put "$repo/artifacts/bfm_teleop_20260917/orin_bench.py" src/orin_bench.py
put "$repo/gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml" data/g1_23dof_rev_1_0.xml
put "$repo/gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json" data/g1_23dof_mujoco_sim2sim.json
for f in "$repo"/gear_sonic/data/robots/g1/meshes/*; do [ -f "$f" ] && put "$f" "data/meshes/$(basename "$f")"; done
put "$repo/artifacts/g1_true23_six_hour_replan_20260910_v1/bfmzero_inference_v1/inference.safetensors" weights/inference.safetensors
put "$repo/artifacts/g1_true23_six_hour_replan_20260910_v1/bfmzero_inspect_v1/config.yaml" weights/config.yaml
put "$repo/artifacts/bfm_teleop_20260917/orin_local_input/data/pico_teleop_50hz.npz" data/pico_teleop_50hz.npz
put "$repo/artifacts/bfm_teleop_20260917/orin_local_input/data/windows_c1_reference.npz" data/windows_c1_reference.npz
put /mnt/e/codex-artifacts/bfm_teleop_20260917/fix4/pico_lowstate_500hz.flatbin data/pico_lowstate_500hz.flatbin
PYTHONPATH=/home/unitree/bfm_orin_bench/src /home/unitree/bfm_orin_bench/venv/bin/python -m py_compile /home/unitree/bfm_orin_bench/src/orin_bench.py
