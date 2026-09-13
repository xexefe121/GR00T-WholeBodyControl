$ErrorActionPreference = 'Continue'
$runRoot = 'E:\codex-artifacts\sonic23_teleop_resume_20260911\student_actual_oracle_control1_resume1001_v1'
$statePath = Join-Path $runRoot 'process_status.json'
$runStart = [DateTime]::UtcNow.ToString('o')
$status = [ordered]@{ kind='durable_same_branch_resume'; state='RUNNING'; powershell_pid=$PID; started_utc=$runStart; next_global_control=1001; original_requested_controls=1569; physics_verdict='pending'; labels_admissible=$false }
$status | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statePath -Encoding UTF8
$runExit = -1
$runError = $null
try {
    $expected = '43ec52d0d50b535319cce09a6ba69a17dc0a481aedb989d08262af9d40af8c96'
    $observed = (Get-FileHash -LiteralPath (Join-Path $runRoot 'frozen_inputs.json') -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($observed -ne $expected) { throw 'Resume frozen receipt changed after review.' }
    & wsl.exe -d Ubuntu-22.04 --cd /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof -- bash /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python -u /mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/student_actual_oracle_control1_resume1001_v1/source_snapshot/resume_actual_student_oracle.py --stage continue 1> (Join-Path $runRoot 'stdout.log') 2> (Join-Path $runRoot 'stderr.log')
    $runExit = $LASTEXITCODE
} catch {
    $runError = $_.Exception.Message
} finally {
    $status.state = 'EXITED'
    $status['ended_utc'] = [DateTime]::UtcNow.ToString('o')
    $status['process_exit_code'] = $runExit
    $status['launcher_error'] = $runError
    $status['physics_verdict'] = 'Read nominal/report.json and independent audit; process exit is not qualification.'
    $status | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statePath -Encoding UTF8
}
exit $runExit
