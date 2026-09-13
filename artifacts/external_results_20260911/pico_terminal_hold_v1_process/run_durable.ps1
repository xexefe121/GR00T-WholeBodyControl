$ErrorActionPreference = 'Stop'
$base = 'E:\codex-artifacts\sonic23_teleop_resume_20260911'
$processDirectory = Join-Path $base 'pico_terminal_hold_v1_process'
$launcher = Join-Path $base 'launch_pico_terminal_hold_v1.ps1'
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
    foreach ($name in @('pico_terminal_hold_v1','pico_terminal_hold_v1.stdout.log','pico_terminal_hold_v1.stderr.log')) {
        if (Test-Path -LiteralPath (Join-Path $base $name)) { throw ('Output exists; refusing duplicate run: ' + $name) }
    }
    @{state='running';pid=$PID;started_utc=$started;launcher=$launcher;requested_controls=250} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $processDirectory 'running.json')
    $shellPath = (Get-Process -Id $PID).Path
    & $shellPath -NoProfile -NonInteractive -File $launcher
    $exitStatus = $LASTEXITCODE
} catch {
    $errorMessage = $_.Exception.ToString()
    Write-Error $errorMessage -ErrorAction Continue
} finally {
    $reportPath = Join-Path $base 'pico_terminal_hold_v1\report.json'
    $result = @{state='exited';pid=$PID;started_utc=$started;finished_utc=[DateTime]::UtcNow.ToString('o');exit_code=$exitStatus;error=$errorMessage;report_exists=(Test-Path -LiteralPath $reportPath);completed=$false;requested_controls=250}
    if ($result.report_exists) {
        $report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
        $result.completed = $report.probe_completed -and $report.completed_controls -eq 250 -and $report.boundary_verified -and $null -eq $report.failure
        $result.completed_controls = $report.completed_controls
        $result.failure = $report.failure
        $result.report_sha256 = Read-TaskSha256 $reportPath
    }
    $temp = Join-Path $processDirectory 'exit_status.tmp.json'
    $result | ConvertTo-Json -Depth 40 | Set-Content -LiteralPath $temp
    Move-Item -LiteralPath $temp -Destination (Join-Path $processDirectory 'exit_status.json')
}
exit $exitStatus
