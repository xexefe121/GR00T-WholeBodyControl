param([Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$LaunchReceiptSha256)
$ErrorActionPreference='Stop'
$stream=[IO.File]::OpenRead((Join-Path $PSScriptRoot 'launch_receipt.json'))
$algorithm=[Security.Cryptography.SHA256]::Create()
try {$actual=([BitConverter]::ToString($algorithm.ComputeHash($stream))).Replace('-','').ToLowerInvariant()}
finally {$stream.Dispose();$algorithm.Dispose()}
if($actual -ne $LaunchReceiptSha256){throw 'Actual audit launch receipt changed.'}
$guard=[IO.File]::Open((Join-Path $PSScriptRoot 'dispatch_started.lock'),[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
$guard.Dispose()
$arguments=@('-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',(Join-Path $PSScriptRoot 'run_audit_durable.ps1'),'-LaunchReceiptSha256',$LaunchReceiptSha256)
$taskProcess=Start-Process -FilePath "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -ArgumentList $arguments -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $PSScriptRoot 'wrapper_stdout.log') -RedirectStandardError (Join-Path $PSScriptRoot 'wrapper_stderr.log')
$handle=$taskProcess.Handle
$record=[ordered]@{utc=[DateTime]::UtcNow.ToString('o');wrapper_pid=$taskProcess.Id;captured_handle_nonzero=($handle -ne [IntPtr]::Zero);launch_receipt_sha256=$actual;exact_arguments=$arguments;automatic_retry=$false;sole_dispatch_owner='root'}
$stream=[IO.File]::Open((Join-Path $PSScriptRoot 'dispatch.json'),[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
try {$bytes=[Text.Encoding]::UTF8.GetBytes(($record|ConvertTo-Json -Depth 5));$stream.Write($bytes,0,$bytes.Length)} finally {$stream.Dispose()}
if($handle -eq [IntPtr]::Zero){throw 'Wrapper handle unavailable; preserve existing attempt.'}
$record|ConvertTo-Json -Compress
