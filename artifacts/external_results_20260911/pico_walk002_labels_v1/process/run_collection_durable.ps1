$ErrorActionPreference = 'Stop'
function Get-TaskHash([string] $Path) {
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    try { return ([System.BitConverter]::ToString($algorithm.ComputeHash([System.IO.File]::ReadAllBytes($Path)))).Replace('-', '').ToLowerInvariant() }
    finally { $algorithm.Dispose() }
}
$processFolder = 'E:\codex-artifacts\sonic23_teleop_resume_20260911\pico_walk002_labels_v1\process'
$receiptPath = Join-Path $processFolder 'launch_receipt.json'
$receipt = Get-Content -Raw -LiteralPath $receiptPath | ConvertFrom-Json
foreach ($entry in $receipt.input_hashes.PSObject.Properties) { if ((Get-TaskHash $entry.Name) -ne $entry.Value) { throw ('Launch input changed: ' + $entry.Name) } }
$lockPath = Join-Path $processFolder 'started.lock'
$lock = [System.IO.File]::Open($lockPath, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
$lock.Close()
$start = [ordered]@{ utc = [DateTime]::UtcNow.ToString('o'); wrapper_pid = $PID; receipt_sha256 = Get-TaskHash $receiptPath; intended_rows = 6847; inference_only = $true }
$start | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $processFolder 'start.json') -Encoding UTF8
$exitCode = 1
$errorText = $null
try {
    $child = Start-Process -FilePath "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $processFolder 'run_collection.ps1')) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $processFolder 'stdout.log') -RedirectStandardError (Join-Path $processFolder 'stderr.log')
    [ordered]@{ child_pid = $child.Id; wrapper_pid = $PID; utc = [DateTime]::UtcNow.ToString('o') } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $processFolder 'child.json') -Encoding UTF8
    $child.WaitForExit()
    $child.Refresh()
    $exitCode = $child.ExitCode
    $postHashes = [ordered]@{}
    foreach ($entry in $receipt.input_hashes.PSObject.Properties) { $actual = Get-TaskHash $entry.Name; $postHashes[$entry.Name] = $actual; if ($actual -ne $entry.Value) { throw ('Postrun input changed: ' + $entry.Name) } }
    $postHashes | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $processFolder 'postrun_hashes.json') -Encoding UTF8
} catch { $errorText = $_.Exception.Message; $exitCode = 1 } finally {
    [ordered]@{ utc = [DateTime]::UtcNow.ToString('o'); exit_code = $exitCode; error = $errorText; intended_rows = 6847 } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $processFolder 'exit.json') -Encoding UTF8
}
exit $exitCode
