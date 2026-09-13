$ErrorActionPreference = 'Stop'
$started = [DateTime]::UtcNow.ToString('o')
$exitStatus = 1
$errorMessage = $null
function Read-Sha([string]$Path) {
    $stream = [IO.File]::OpenRead($Path)
    $hash = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($hash.ComputeHash($stream))).Replace('-','').ToLowerInvariant() }
    finally { $hash.Dispose(); $stream.Dispose() }
}
try {
    $receipt = Get-Content -Raw -LiteralPath 'E:\codex-artifacts\sonic23_teleop_resume_20260911\walk002_qualified_package_v1\replay_process\launch_receipt.json' | ConvertFrom-Json
    foreach ($entry in $receipt.hashes.PSObject.Properties) { if ((Read-Sha $entry.Name) -ne $entry.Value) { throw ('Input changed: ' + $entry.Name) } }
    if (Test-Path -LiteralPath 'E:\codex-artifacts\sonic23_teleop_resume_20260911\walk002_one_command_replay_smoke_v1') { throw 'Existing output; refusing duplicate execution' }
    @{state='running';pid=$PID;started_utc=$started} | ConvertTo-Json | Set-Content -LiteralPath 'E:\codex-artifacts\sonic23_teleop_resume_20260911\walk002_qualified_package_v1\replay_process\running.json'
    & 'E:\codex-artifacts\sonic23_teleop_resume_20260911\walk002_qualified_package_v1\RUN_WALK002_RECORDED_PHYSICS_REPLAY.ps1' -RunName 'walk002_one_command_replay_smoke_v1'
    $exitStatus = $LASTEXITCODE
} catch { $errorMessage = $_.Exception.ToString(); Write-Error $errorMessage -ErrorAction Continue }
finally {
    $result = @{pid=$PID;started_utc=$started;finished_utc=[DateTime]::UtcNow.ToString('o');exit_code=$exitStatus;error=$errorMessage;result_exists=(Test-Path -LiteralPath 'E:\codex-artifacts\sonic23_teleop_resume_20260911\walk002_one_command_replay_smoke_v1\replay_result.json')}
    if (-not $result.result_exists) { $exitStatus=1; $result.exit_code=1 }
    if ($result.result_exists) { $result.result_sha256=Read-Sha 'E:\codex-artifacts\sonic23_teleop_resume_20260911\walk002_one_command_replay_smoke_v1\replay_result.json' }
    $result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath 'E:\codex-artifacts\sonic23_teleop_resume_20260911\walk002_qualified_package_v1\replay_process\exit_status.json'
}
exit $exitStatus
