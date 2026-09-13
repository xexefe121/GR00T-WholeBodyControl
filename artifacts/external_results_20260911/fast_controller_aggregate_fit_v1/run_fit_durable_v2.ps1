$ErrorActionPreference = 'Continue'
$runRoot = 'E:\codex-artifacts\sonic23_teleop_resume_20260911\fast_controller_aggregate_fit_v1'
$statePath = Join-Path $runRoot 'fit_process_status.json'
foreach ($existing in @('fit_process_status.json', 'fit_stdout.log', 'fit_stderr.log', 'fit\request.json')) {
    if (Test-Path -LiteralPath (Join-Path $runRoot $existing)) { throw 'Existing fit attempt must remain preserved; do not relaunch.' }
}
$status = [ordered]@{ kind='single_aggregate_fit'; state='TRAINING'; powershell_pid=$PID; started_utc=[DateTime]::UtcNow.ToString('o'); first_global_step=20001; last_global_step=60000; additional_updates=40000; physics_steps=0; canonical_rollout_started=$false }
$status | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statePath -Encoding UTF8
$runExit = -1
$runError = $null
try {
    $digest = [Security.Cryptography.SHA256]::Create()
    try { $observed = ([BitConverter]::ToString($digest.ComputeHash([IO.File]::ReadAllBytes((Join-Path $runRoot 'frozen_inputs_v2.json'))))).Replace('-', '').ToLowerInvariant() } finally { $digest.Dispose() }
    if ($observed -ne '1471e7fd28602e829eed1ac14bae9d3e8474f5a1bdaeeeedb6bdffbf0bdd3fb7') { throw 'Reviewed aggregate fit receipt changed.' }
    $env:OMP_NUM_THREADS = '1'
    $env:OPENBLAS_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    & 'C:\Users\camer\AppData\Local\Programs\Python\Python310\python.exe' -u (Join-Path $runRoot 'source_snapshot_v2\fit_aggregate_once.py') 1> (Join-Path $runRoot 'fit_stdout.log') 2> (Join-Path $runRoot 'fit_stderr.log')
    $runExit = $LASTEXITCODE
} catch {
    $runError = $_.Exception.Message
} finally {
    $status.state = 'EXITED'
    $status['ended_utc'] = [DateTime]::UtcNow.ToString('o')
    $status['process_exit_code'] = $runExit
    $status['launcher_error'] = $runError
    $status['next_gate'] = 'Review ordinary final export hashes and parity before the one canonical lifecycle.'
    $status | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statePath -Encoding UTF8
}
exit $runExit
