$ErrorActionPreference = 'Stop'
$runBase = 'E:\codex-artifacts\sonic23_teleop_resume_20260911'
$wslArguments = @(
    '-d', 'Ubuntu-22.04', '--cd', '/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof',
    '--', 'bash', '/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh',
    'env', 'PYTHONPATH=/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/pico_control_lm_integration_v1/repo',
    'OPENBLAS_NUM_THREADS=1', 'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
    '/mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python',
    '/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/pico_control_lm_integration_v1/repo/gear_sonic/scripts/evaluate_g1_true23_mjbatch_mpc.py',
    '--bundle', 'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1',
    '--clip', 'walk002', '--probe', 'full-lifecycle', '--horizon', '30', '--iterations', '5',
    '--commit', '5', '--threads', '8', '--fd-epsilon', '1e-6', '--feedback-clip', '.1',
    '--all-joint-limit-margin', '.05', '--all-joint-limit-weight', '2000', '--relative-foot-weight', '400',
    '--hard-feasibility', '--restoration', '--restoration-control-lm-retry',
    '--target-seed', 'artifacts/teleop_six_hour_20260910/bfm_walk002_feedback_v2',
    '--fresh-bfm-seed-onnx', 'artifacts/teleop_six_hour_20260910/bfm_onnx_v2',
    '--fresh-bfm-seed-dependencies', '/mnt/e/codex_sonic_runtime/bfm_seed_20260910/onnx_deps',
    '--motion-override', '/mnt/e/codex-artifacts/sonic23_teleop_six_hour_20260910/mjbatch_intent_floor_inputs_v1/walk002/reference.npz',
    '--checkpoint-controls', '100',
    '--output', '/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/walk002_full_control_lm_v1'
)
& wsl.exe @wslArguments 1> (Join-Path $runBase 'walk002_full_control_lm_v1.stdout.log') 2> (Join-Path $runBase 'walk002_full_control_lm_v1.stderr.log')
exit $LASTEXITCODE
