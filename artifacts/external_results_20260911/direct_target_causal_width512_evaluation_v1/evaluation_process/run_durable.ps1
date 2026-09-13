$ErrorActionPreference = 'Stop'
function Get-TaskHash([string] $Path) {
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    $stream = [System.IO.File]::Open($Path,[System.IO.FileMode]::Open,[System.IO.FileAccess]::Read,([System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete))
    try { return ([System.BitConverter]::ToString($algorithm.ComputeHash($stream))).Replace('-', '').ToLowerInvariant() }
    finally { $stream.Dispose(); $algorithm.Dispose() }
}
function Write-NewJson([string] $Path, $Value) {
    $text = $Value | ConvertTo-Json -Depth 12
    $stream = [System.IO.File]::Open($Path,[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write,[System.IO.FileShare]::Read)
    try { $bytes = [System.Text.Encoding]::UTF8.GetBytes($text); $stream.Write($bytes,0,$bytes.Length) } finally { $stream.Dispose() }
}
$folder = 'E:\codex-artifacts\sonic23_teleop_resume_20260911\direct_target_causal_width512_evaluation_v1\evaluation_process'
$receiptPath = Join-Path $folder 'launch_receipt.json'
$receipt = Get-Content -Raw -LiteralPath $receiptPath | ConvertFrom-Json
$clearancePath = Join-Path $folder 'launch_clearance.json'
$clearance = Get-Content -Raw -LiteralPath $clearancePath | ConvertFrom-Json
if ($clearance.launch_receipt_sha256 -ne (Get-TaskHash $receiptPath) -or -not $clearance.selected_single_run) { throw 'Actual final launch clearance missing or changed.' }
if ((Get-TaskHash $clearance.review.path) -ne $clearance.review.sha256) { throw 'Final reviewer receipt changed.' }
$review = Get-Content -Raw -LiteralPath $clearance.review.path | ConvertFrom-Json
$pass = $review
foreach ($key in $clearance.review.pass_field.Split('.')) { $pass = $pass.$key }
if ($pass -ne $true) { throw 'Final reviewer gate not passed.' }
if ($review.launch_receipt_subject.sha256 -ne $clearance.launch_receipt_sha256 -or $review.launch_receipt_subject.path -ne $receiptPath.Replace([char]92,[char]47) -or $review.binding_subject.sha256 -ne $receipt.binding_sha256 -or $review.binding_subject.path -ne $receipt.binding_path) { throw 'Final review does not name this exact binding and launch.' }
foreach ($entry in $receipt.input_hashes.PSObject.Properties) { if ((Get-TaskHash $entry.Name) -ne $entry.Value) { throw ('Launch input changed: ' + $entry.Name) } }
$lock = [System.IO.File]::Open((Join-Path $folder 'started.lock'),[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write,[System.IO.FileShare]::None)
$lock.Close()
Write-NewJson (Join-Path $folder 'start.json') ([ordered]@{utc=[DateTime]::UtcNow.ToString('o');wrapper_pid=$PID;receipt_sha256=(Get-TaskHash $receiptPath);clearance_sha256=(Get-TaskHash $clearancePath);review_sha256=(Get-TaskHash $clearance.review.path)})
$exitCode = 1
$childExit = $null
$errorText = $null
try {
    $taskChild = Start-Process -FilePath "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',(Join-Path $folder 'run.ps1')) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $folder 'stdout.log') -RedirectStandardError (Join-Path $folder 'stderr.log')
    $nativeHandle = $taskChild.Handle
    Write-NewJson (Join-Path $folder 'child.json') ([ordered]@{child_pid=$taskChild.Id;wrapper_pid=$PID;handle_acquired=($nativeHandle -ne [IntPtr]::Zero)})
    $taskChild.WaitForExit()
    $childExit = $taskChild.ExitCode
    if ($null -eq $childExit) { throw 'Child exit status unavailable.' }
    $exitCode = $childExit
} catch { $errorText=$_.Exception.Message; $exitCode=1 } finally {
    $post = [ordered]@{}
    $allExact = $true
    foreach ($entry in $receipt.input_hashes.PSObject.Properties) {
        try { $actual = Get-TaskHash $entry.Name } catch { $actual = $null }
        $post[$entry.Name] = $actual
        if ($actual -ne $entry.Value) { $allExact=$false }
    }
    Write-NewJson (Join-Path $folder 'postrun_hashes.json') $post
    if (-not $allExact) { $exitCode=1; if ($null -eq $errorText) { $errorText='One or more postrun inputs changed.' } }
    if ((Get-TaskHash $clearancePath) -ne (Get-Content -Raw -LiteralPath (Join-Path $folder 'start.json') | ConvertFrom-Json).clearance_sha256) { $exitCode=1; $errorText='Launch clearance changed.' }
    Write-NewJson (Join-Path $folder 'exit.json') ([ordered]@{utc=[DateTime]::UtcNow.ToString('o');exit_code=$exitCode;raw_child_exit_code=$childExit;error=$errorText;all_postrun_hashes_exact=$allExact;automatic_retry=$false})
}
exit $exitCode
