param([Parameter(Mandatory=$true)][ValidateSet('witness','evaluation')][string]$Mode,
      [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$ClearanceSha256)
$ErrorActionPreference='Stop'
$folder=Join-Path $PSScriptRoot ($Mode+'_process')
$algorithm=[Security.Cryptography.SHA256]::Create()
$stream=[IO.File]::OpenRead((Join-Path $folder 'launch_clearance.json'))
try {$actual=([BitConverter]::ToString($algorithm.ComputeHash($stream))).Replace('-','').ToLowerInvariant()}
finally {$stream.Dispose();$algorithm.Dispose()}
if($actual -ne $ClearanceSha256){throw 'Actual clearance changed.'}
$guard=[IO.File]::Open((Join-Path $folder 'dispatch_started.lock'),[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
$guard.Close()
$arguments=@('-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',(Join-Path $folder 'run_durable.ps1'))
$taskProcess=Start-Process -FilePath "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -ArgumentList $arguments -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $folder 'wrapper_stdout.log') -RedirectStandardError (Join-Path $folder 'wrapper_stderr.log')
$handle=$taskProcess.Handle
$record=[ordered]@{utc=[DateTime]::UtcNow.ToString('o');wrapper_pid=$taskProcess.Id;captured_handle_nonzero=($handle -ne [IntPtr]::Zero);clearance_sha256=$actual;exact_arguments=$arguments;automatic_retry=$false;sole_dispatch_owner='root'}
$stream=[IO.File]::Open((Join-Path $folder 'dispatch.json'),[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
try {$bytes=[Text.Encoding]::UTF8.GetBytes(($record|ConvertTo-Json -Depth 5));$stream.Write($bytes,0,$bytes.Length)} finally {$stream.Dispose()}
if($handle -eq [IntPtr]::Zero){throw 'Wrapper handle unavailable; preserve existing attempt.'}
$record|ConvertTo-Json -Compress
