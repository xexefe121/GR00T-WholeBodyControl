# Reproduce the current offline PICO candidate

This is the exact PowerShell-to-WSL invocation used on 2026-09-10 at18:36UTC, with only the output directory changed to a clearly new path. It runs offline simulation and writes a new result. No robot, DDS or live PICO transport is involved. Existing output directories are rejected.

```powershell
wsl -d Ubuntu-22.04 --cd /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof -- env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python -m gear_sonic.scripts.evaluate_g1_true23_mjbatch_mpc --bundle artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1 --clip pico --probe full-lifecycle --horizon 30 --commit 5 --iterations 5 --threads 8 --feedback-clip 0.1 --checkpoint-controls 100 --relative-foot-weight 400 --all-joint-limit-margin 0.05 --all-joint-limit-weight 2000 --target-seed artifacts/teleop_six_hour_20260910/bfm_pico_feedback_v2 --motion-override /mnt/e/codex-artifacts/sonic23_teleop_six_hour_20260910/mjbatch_intent_floor_inputs_v1/pico/reference.npz --fresh-bfm-seed-onnx artifacts/teleop_six_hour_20260910/bfm_onnx_v2 --fresh-bfm-seed-dependencies /mnt/e/codex_sonic_runtime/bfm_seed_20260910/onnx_deps --output /mnt/e/codex-artifacts/sonic23_teleop_six_hour_20260910/mjbatch_full_v1/REPRODUCE_pico_fresh_allmargin_NEW_OUTPUT
```

The runtime is isolated MuJoCo3.2.3/mjbatch with NumPy1.26.4. The invocation does not set a `PYTHONPATH` override: `--cd` selects the repository and Python's `-m` module execution resolves its packages there. The fresh-seed helper explicitly loads ONNX Runtime from the dependency directory passed above. It binds the actual imported runtime and graph hashes in `request.json`.

The original BFM goal motion is `artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/pico/native_original.npz`, loaded from the bundle before the motion override. The recorded BFM seed supplies only its previously applied targets. Each fresh seed instead starts from the current measured simulated state and actual applied-target history; it uses the original BFM goal with position gain1, yaw gain2 and eight goal packets. The optimizer uses the explicitly validated v4 floor-adjusted motion override. No seed physical trajectory is copied into the plant.

The control clock is50Hz and native PD physics remains500Hz. H30 and the BFM seed imply0.74s supplied-packet support and a conservative0.76s raw-pose support bound. This is not the140ms received-only stream controller. Finite-difference epsilon remains the default1e-6; the common joint margin is a soft predicted-state penalty. Every2ms native range, speed, effort, warnings and cumulative clock still require validation.

The executed result directory is `E:\codex-artifacts\sonic23_teleop_six_hour_20260910\mjbatch_full_v1\pico_v4_native323_freshseed_allmargin_5iter_full_v1`. Its immutable request and source snapshots define the exact executed version. The command above resolves current working-tree modules; compare their hashes to those snapshots before claiming a byte-identical reproduction.

After a fresh WSL restart, if E: is not mounted, mount it once before running the command:

```powershell
wsl -d Ubuntu-22.04 -- bash -c 'if ! mountpoint -q /mnt/e; then mount -t drvfs E: /mnt/e; fi'
```
