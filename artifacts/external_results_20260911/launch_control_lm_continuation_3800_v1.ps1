& wsl -d Ubuntu-22.04 --cd /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof -- bash -c @'
if ! mountpoint -q /mnt/e; then mount -t drvfs E: /mnt/e; fi
exec env PYTHONPATH=/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/control_lm_continuation_3800_v1_source/repo OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python /mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/control_lm_continuation_3800_v1_source/repo/gear_sonic/scripts/continue_g1_true23_mpc_hard_feasibility.py \
  --bundle artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1 \
  --fixture /mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/control_lm_continuation_3800_v1_source/input/actual3800_fixture.npz \
  --history /mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/control_lm_continuation_3800_v1_source/input/actual3800_history.npz \
  --motion-override /mnt/e/codex-artifacts/sonic23_teleop_six_hour_20260910/mjbatch_intent_floor_inputs_v1/pico/reference.npz \
  --recorded-seed artifacts/teleop_six_hour_20260910/bfm_pico_feedback_v2 \
  --onnx artifacts/teleop_six_hour_20260910/bfm_onnx_v2 \
  --dependencies /mnt/e/codex_sonic_runtime/bfm_seed_20260910/onnx_deps \
  --first-proposal-witnesses /mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/pico_full_hard_restoration_v1 \
  --control-lm-witness /mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/pico_final3800_control_lm_k0_v1/ordinary_final.npz \
  --canonical-trace /mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/pico_full_hard_restoration_v1/trace.npz \
  --controls 100 --restoration --restoration-zero-feedback-retry --restoration-control-lm-retry \
  --output /mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/control_lm_continuation_3800_v1 \
  > /mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/control_lm_continuation_3800_v1.stdout.log 2>&1
'@
exit $LASTEXITCODE
