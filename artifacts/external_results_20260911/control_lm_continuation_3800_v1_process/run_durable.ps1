$ErrorActionPreference = 'Stop'
$base = 'E:\codex-artifacts\sonic23_teleop_resume_20260911'
$processDirectory = Join-Path $base 'control_lm_continuation_3800_v1_process'
$launcher = Join-Path $base 'launch_control_lm_continuation_3800_v1.ps1'
$started = [DateTime]::UtcNow.ToString('o')
$exitStatus = 1
$errorMessage = $null
try {
    $receipt = Get-Content -LiteralPath (Join-Path $processDirectory 'launch_receipt.json') -Raw | ConvertFrom-Json
    foreach ($entry in $receipt.verified_hashes.PSObject.Properties) {
        if ((Get-FileHash -LiteralPath $entry.Name -Algorithm SHA256).Hash.ToLower() -ne $entry.Value) { throw ('Input hash mismatch: ' + $entry.Name) }
    }
    if (Test-Path -LiteralPath (Join-Path $base 'control_lm_continuation_3800_v1')) { throw 'Output already exists; refusing duplicate run' }
    if (Test-Path -LiteralPath (Join-Path $base 'control_lm_continuation_3800_v1.stdout.log')) { throw 'Run log already exists; refusing duplicate run' }
    @{state='running';pid=$PID;started_utc=$started;launcher=$launcher} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $processDirectory 'running.json')
    $shellPath = (Get-Process -Id $PID).Path
    & $shellPath -NoProfile -NonInteractive -File $launcher
    $exitStatus = $LASTEXITCODE
} catch {
    $errorMessage = $_.Exception.ToString()
    Write-Error $errorMessage -ErrorAction Continue
} finally {
    $reportPath = Join-Path $base 'control_lm_continuation_3800_v1\report.json'
    $result = @{state='exited';pid=$PID;started_utc=$started;finished_utc=[DateTime]::UtcNow.ToString('o');exit_code=$exitStatus;error=$errorMessage;report_exists=(Test-Path -LiteralPath $reportPath);completed=$false}
    if ($result.report_exists) {
        $report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
        $result.completed = $report.completed
        $result.completed_controls = $report.completed_controls
        $result.failure = $report.failure
        $result.report_sha256 = (Get-FileHash -LiteralPath $reportPath -Algorithm SHA256).Hash.ToLower()
    }
    $temp = Join-Path $processDirectory 'exit_status.tmp.json'
    $result | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $temp
    Move-Item -LiteralPath $temp -Destination (Join-Path $processDirectory 'exit_status.json')
}
exit $exitStatus
