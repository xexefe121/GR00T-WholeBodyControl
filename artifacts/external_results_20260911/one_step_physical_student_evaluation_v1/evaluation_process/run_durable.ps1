$ErrorActionPreference = 'Stop'
function Get-TaskHash([string] $Path) {
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    try { return ([System.BitConverter]::ToString($algorithm.ComputeHash([System.IO.File]::ReadAllBytes($Path)))).Replace('-', '').ToLowerInvariant() }
    finally { $algorithm.Dispose() }
}
$folder = 'E:\codex-artifacts\sonic23_teleop_resume_20260911\one_step_physical_student_evaluation_v1\evaluation_process'
$receiptPath = Join-Path $folder 'launch_receipt.json'
$receipt = Get-Content -Raw -LiteralPath $receiptPath | ConvertFrom-Json
foreach ($entry in $receipt.input_hashes.PSObject.Properties) { if ((Get-TaskHash $entry.Name) -ne $entry.Value) { throw ('Launch input changed: ' + $entry.Name) } }
$lock = [System.IO.File]::Open((Join-Path $folder 'started.lock'),[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write,[System.IO.FileShare]::None)
$lock.Close()
[ordered]@{utc=[DateTime]::UtcNow.ToString('o');wrapper_pid=$PID;receipt_sha256=(Get-TaskHash $receiptPath)} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $folder 'start.json') -Encoding UTF8
$exitCode = 1
$errorText = $null
try {
    $taskChild = Start-Process -FilePath "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',(Join-Path $folder 'run.ps1')) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $folder 'stdout.log') -RedirectStandardError (Join-Path $folder 'stderr.log')
    $nativeHandle = $taskChild.Handle
    [ordered]@{child_pid=$taskChild.Id;wrapper_pid=$PID;handle_acquired=($nativeHandle -ne [IntPtr]::Zero)} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $folder 'child.json') -Encoding UTF8
    $taskChild.WaitForExit()
    $exitCode = $taskChild.ExitCode
    if ($null -eq $exitCode) { throw 'Child exit status unavailable.' }
    $post = [ordered]@{}
    foreach ($entry in $receipt.input_hashes.PSObject.Properties) { $actual = Get-TaskHash $entry.Name; $post[$entry.Name] = $actual; if ($actual -ne $entry.Value) { throw ('Postrun input changed: ' + $entry.Name) } }
    $post | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $folder 'postrun_hashes.json') -Encoding UTF8
} catch { $errorText=$_.Exception.Message; $exitCode=1 } finally {
    [ordered]@{utc=[DateTime]::UtcNow.ToString('o');exit_code=$exitCode;error=$errorText} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $folder 'exit.json') -Encoding UTF8
}
exit $exitCode
