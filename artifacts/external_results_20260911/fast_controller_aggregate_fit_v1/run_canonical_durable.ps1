param([Parameter(Mandatory=$true)][string]$ClearanceSha256)
$ErrorActionPreference = 'Continue'
$runRoot = 'E:\codex-artifacts\sonic23_teleop_resume_20260911\fast_controller_aggregate_fit_v1'
foreach ($existing in @('canonical_process_status.json', 'canonical_stdout.log', 'canonical_stderr.log', 'nominal', 'post_lifecycle_hold_5s')) {
    if (Test-Path -LiteralPath (Join-Path $runRoot $existing)) { throw 'Existing canonical attempt must remain preserved; do not relaunch.' }
}
function Read-Sha256([string]$path) {
    $digest = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($digest.ComputeHash([IO.File]::ReadAllBytes($path)))).Replace('-', '').ToLowerInvariant() } finally { $digest.Dispose() }
}
$clearancePath = Join-Path $runRoot 'final_export_clearance.json'
if ((Read-Sha256 $clearancePath) -ne $ClearanceSha256) { throw 'Final export clearance identity mismatch.' }
$clearance = Get-Content -LiteralPath $clearancePath -Raw | ConvertFrom-Json
if ($clearance.canonical_rollout_authorized -ne $true -or $clearance.ordinary_final_global_step -ne 60000 -or $clearance.additional_updates -ne 40000) { throw 'Final export clearance does not authorize this canonical rollout.' }
if ($clearance.export_parity_passed -ne $true) { throw 'Final export parity has not passed.' }
foreach ($required in @('fit\student_head.onnx', 'fit\student_head.pt', 'fit\report.json', 'fit\restoration20000_parity.json', 'fit\optimization_completed.json', 'frozen_inputs_v2.json', 'run_canonical_durable.ps1')) {
    $requiredPath = [IO.Path]::GetFullPath((Join-Path $runRoot $required))
    $bindings = @($clearance.bound_files | Where-Object { [IO.Path]::GetFullPath($_.path) -ieq $requiredPath })
    if ($bindings.Count -ne 1) { throw ('Required final input not bound exactly once: ' + $required) }
}
foreach ($item in $clearance.bound_files) {
    if ((Read-Sha256 $item.path) -ne $item.sha256) { throw ('Final reviewed input changed: ' + $item.path) }
}
if ((Read-Sha256 (Join-Path $runRoot 'frozen_inputs_v2.json')) -ne '1471e7fd28602e829eed1ac14bae9d3e8474f5a1bdaeeeedb6bdffbf0bdd3fb7') { throw 'Frozen aggregate source changed.' }
$fit = Get-Content -LiteralPath (Join-Path $runRoot 'fit\report.json') -Raw | ConvertFrom-Json
if ($fit.network_training_complete -ne $true -or $fit.steps -ne 60000 -or $fit.additional_updates -ne 40000 -or $fit.ONNX_vs_Torch_max_delta_rad -ge 0.00001) { throw 'Ordinary final fit/export report did not pass.' }
$statePath = Join-Path $runRoot 'canonical_process_status.json'
$status = [ordered]@{ kind='one_canonical_aggregate_student_rollout'; state='RUNNING'; powershell_pid=$PID; started_utc=[DateTime]::UtcNow.ToString('o'); requested_controls=1569; separate_hold_controls=250; student_generates_control0=$true; cached_expert_commands_used=$false; hardware_authorized=$false; clearance_sha256=$ClearanceSha256 }
$status | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statePath -Encoding UTF8
$runExit = -1
$runError = $null
try {
    & wsl.exe -d Ubuntu-22.04 --cd /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof -- bash /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python -u /mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/fast_controller_aggregate_fit_v1/source_snapshot_v2/evaluate_nominal_pilot.py 1> (Join-Path $runRoot 'canonical_stdout.log') 2> (Join-Path $runRoot 'canonical_stderr.log')
    $runExit = $LASTEXITCODE
} catch {
    $runError = $_.Exception.Message
} finally {
    $status.state = 'EXITED'
    $status['ended_utc'] = [DateTime]::UtcNow.ToString('o')
    $status['process_exit_code'] = $runExit
    $status['launcher_error'] = $runError
    $status['physical_verdict'] = 'Read nominal/report.json, separate hold and independent audits; process exit is not qualification.'
    $status | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statePath -Encoding UTF8
}
exit $runExit
