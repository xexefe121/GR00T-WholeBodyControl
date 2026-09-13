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
    $receipt = Get-Content -Raw -LiteralPath 'E:\codex-artifacts\sonic23_teleop_resume_20260911\walk002_qualified_package_v1\render_process\launch_receipt.json' | ConvertFrom-Json
    foreach ($entry in $receipt.hashes.PSObject.Properties) { if ((Read-Sha $entry.Name) -ne $entry.Value) { throw ('Input changed: ' + $entry.Name) } }
    if (Test-Path -LiteralPath 'E:\codex-artifacts\sonic23_teleop_resume_20260911\walk002_qualified_hybrid_video_v1') { throw 'Existing output; refusing duplicate execution' }
    @{state='running';pid=$PID;started_utc=$started} | ConvertTo-Json | Set-Content -LiteralPath 'E:\codex-artifacts\sonic23_teleop_resume_20260911\walk002_qualified_package_v1\render_process\running.json'
    & 'C:/Users/camer/AppData/Local/Programs/Python/Python310/python.exe' 'E:\codex-artifacts\sonic23_teleop_resume_20260911\walk002_qualified_package_v1\render_walk002_qualified.py' --qualification 'E:\codex-artifacts\sonic23_teleop_resume_20260911\walk002_hybrid_root_qualification_v1\qualification.json' --qualification-sha256 'a7eaec31dd572724921e2d2986ab8c6ab5db33008f6b3c1af5d2dbd72bf92e4b'
    $exitStatus = $LASTEXITCODE
} catch { $errorMessage = $_.Exception.ToString(); Write-Error $errorMessage -ErrorAction Continue }
finally {
    $result = @{pid=$PID;started_utc=$started;finished_utc=[DateTime]::UtcNow.ToString('o');exit_code=$exitStatus;error=$errorMessage;result_exists=(Test-Path -LiteralPath 'E:\codex-artifacts\sonic23_teleop_resume_20260911\walk002_qualified_hybrid_video_v1\render_receipt.json')}
    if (-not $result.result_exists) { $exitStatus=1; $result.exit_code=1 }
    if ($result.result_exists) { $result.result_sha256=Read-Sha 'E:\codex-artifacts\sonic23_teleop_resume_20260911\walk002_qualified_hybrid_video_v1\render_receipt.json' }
    $result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath 'E:\codex-artifacts\sonic23_teleop_resume_20260911\walk002_qualified_package_v1\render_process\exit_status.json'
}
exit $exitStatus
