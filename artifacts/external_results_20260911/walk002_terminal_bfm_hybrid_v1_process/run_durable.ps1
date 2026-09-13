$ErrorActionPreference = 'Stop'
$base = 'E:\codex-artifacts\sonic23_teleop_resume_20260911'
$processDirectory = Join-Path $base 'walk002_terminal_bfm_hybrid_v1_process'
$launcher = Join-Path $base 'launch_walk002_terminal_bfm_hybrid_v1.ps1'
$started = [DateTime]::UtcNow.ToString('o')
$exitStatus = 1
$errorMessage = $null
function Read-TaskSha256([string]$Path) {
    $stream = [System.IO.File]::OpenRead($Path)
    $hasher = [System.Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($hasher.ComputeHash($stream))).Replace('-','').ToLowerInvariant() }
    finally { $hasher.Dispose(); $stream.Dispose() }
}
try {
    $receipt = Get-Content -LiteralPath (Join-Path $processDirectory 'launch_receipt.json') -Raw | ConvertFrom-Json
    foreach ($entry in $receipt.verified_hashes.PSObject.Properties) {
        if ((Read-TaskSha256 $entry.Name) -ne $entry.Value) { throw ('Input hash mismatch: ' + $entry.Name) }
    }
    foreach ($name in @('walk002_terminal_bfm_hybrid_v1','walk002_terminal_bfm_hybrid_v1.stdout.log','walk002_terminal_bfm_hybrid_v1.stderr.log')) {
        if (Test-Path -LiteralPath (Join-Path $base $name)) { throw ('Output exists; refusing duplicate run: ' + $name) }
    }
    @{state='running';pid=$PID;started_utc=$started;launcher=$launcher;requested_controls=1417;separate_hold_controls=250} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $processDirectory 'running.json')
    $shellPath = (Get-Process -Id $PID).Path
    & $shellPath -NoProfile -NonInteractive -File $launcher
    $exitStatus = $LASTEXITCODE
} catch {
    $errorMessage = $_.Exception.ToString()
    Write-Error $errorMessage -ErrorAction Continue
} finally {
    $comparisonPath = Join-Path $base 'walk002_terminal_bfm_hybrid_v1\comparison.json'
    $result = @{state='exited';pid=$PID;started_utc=$started;finished_utc=[DateTime]::UtcNow.ToString('o');exit_code=$exitStatus;error=$errorMessage;comparison_exists=(Test-Path -LiteralPath $comparisonPath);completed=$false;requested_controls=1417;separate_hold_requested_controls=250}
    if ($result.comparison_exists) {
        $report = Get-Content -LiteralPath $comparisonPath -Raw | ConvertFrom-Json
        $result.physical_completed = $report.lifecycle.completed -and $null -ne $report.extension -and $report.extension.completed
        $result.main_quiet_pass = $report.lifecycle.quiet_pass
        $result.hold_quiet_pass = $null -ne $report.extension -and $report.extension.quiet_pass
        $result.completed = $result.physical_completed -and $result.main_quiet_pass -and $result.hold_quiet_pass -and $report.inputs_unchanged
        if (-not $result.completed) { $exitStatus = 1; $result.exit_code = 1 }
        $result.completed_controls = $report.lifecycle.completed_controls
        $result.hold_completed_controls = $report.extension.completed_controls
        $result.failure = $report.lifecycle.failure
        $result.comparison_sha256 = Read-TaskSha256 $comparisonPath
    }
    $temp = Join-Path $processDirectory 'exit_status.tmp.json'
    $result | ConvertTo-Json -Depth 40 | Set-Content -LiteralPath $temp
    Move-Item -LiteralPath $temp -Destination (Join-Path $processDirectory 'exit_status.json')
}
exit $exitStatus
