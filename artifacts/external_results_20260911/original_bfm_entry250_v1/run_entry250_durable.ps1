param([Parameter(Mandatory=$true)][string]$ClearanceSha256)
$ErrorActionPreference = 'Continue'
$runRoot = 'E:\codex-artifacts\sonic23_teleop_resume_20260911\original_bfm_entry250_v1'
foreach ($existing in @('process_status.json', 'stdout.log', 'stderr.log', 'entry250')) {
    if (Test-Path -LiteralPath (Join-Path $runRoot $existing)) { throw 'Existing baseline attempt must remain preserved; do not relaunch.' }
}
function Read-Sha256([string]$path) {
    $digest = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($digest.ComputeHash([IO.File]::ReadAllBytes($path)))).Replace('-', '').ToLowerInvariant() } finally { $digest.Dispose() }
}
$clearancePath = Join-Path $runRoot 'prelaunch_clearance.json'
if ((Read-Sha256 $clearancePath) -ne $ClearanceSha256) { throw 'Baseline clearance identity mismatch.' }
$clearance = Get-Content -LiteralPath $clearancePath -Raw | ConvertFrom-Json
if ($clearance.one_baseline_run_authorized -ne $true -or $clearance.requested_controls -ne 250 -or $clearance.residual_disabled -ne $true) { throw 'Clearance does not authorize the bounded original BFM baseline.' }
foreach ($required in @('frozen_inputs_v2.json', 'run_entry250_durable.ps1')) {
    $requiredPath = [IO.Path]::GetFullPath((Join-Path $runRoot $required))
    $bindings = @($clearance.bound_files | Where-Object { [IO.Path]::GetFullPath($_.path) -ieq $requiredPath })
    if ($bindings.Count -ne 1) { throw ('Required reviewed input not bound exactly once: ' + $required) }
}
foreach ($item in $clearance.bound_files) {
    if ((Read-Sha256 $item.path) -ne $item.sha256) { throw ('Reviewed baseline input changed: ' + $item.path) }
}
$statePath = Join-Path $runRoot 'process_status.json'
$status = [ordered]@{ kind='one_original_BFM_canonical_entry250'; state='RUNNING'; powershell_pid=$PID; started_utc=[DateTime]::UtcNow.ToString('o'); requested_controls=250; residual_disabled=$true; physical_state_resets_after_initialization=0; hardware_authorized=$false; clearance_sha256=$ClearanceSha256 }
$status | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statePath -Encoding UTF8
$runExit = -1
$runError = $null
try {
    & wsl.exe -d Ubuntu-22.04 --cd /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof -- bash /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python -u /mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/original_bfm_entry250_v1/source_snapshot_v1/evaluate_entry250.py 1> (Join-Path $runRoot 'stdout.log') 2> (Join-Path $runRoot 'stderr.log')
    $runExit = $LASTEXITCODE
} catch {
    $runError = $_.Exception.Message
} finally {
    $status.state = 'EXITED'
    $status['ended_utc'] = [DateTime]::UtcNow.ToString('o')
    $status['process_exit_code'] = $runExit
    $status['launcher_error'] = $runError
    $status['physical_verdict'] = 'Read entry250/report.json and prefix100 parity, then independent audits; process exit is not qualification.'
    $status | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statePath -Encoding UTF8
}
exit $runExit
