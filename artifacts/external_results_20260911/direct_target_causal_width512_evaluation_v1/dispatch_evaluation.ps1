$ErrorActionPreference='Stop'
$folder=Join-Path $PSScriptRoot 'evaluation_process'
$expected='b83640bc15fc8c3cf67e779a07eb9a926eb53c04a29504267c5e1f376be10f38'
$algorithm=[System.Security.Cryptography.SHA256]::Create()
$stream=[System.IO.File]::OpenRead((Join-Path $folder 'launch_clearance.json'))
try {$actual=([System.BitConverter]::ToString($algorithm.ComputeHash($stream))).Replace('-','').ToLowerInvariant()}
finally {$stream.Dispose();$algorithm.Dispose()}
if($actual -ne $expected){throw 'Canonical clearance changed.'}
$guard=[System.IO.File]::Open((Join-Path $folder 'dispatch_started.lock'),[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write,[System.IO.FileShare]::None)
$guard.Close()
$arguments=@('-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',(Join-Path $folder 'run_durable.ps1'))
$taskProcess=Start-Process -FilePath "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -ArgumentList $arguments -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $folder 'wrapper_stdout.log') -RedirectStandardError (Join-Path $folder 'wrapper_stderr.log')
$handle=$taskProcess.Handle
$record=[ordered]@{utc=[DateTime]::UtcNow.ToString('o');wrapper_pid=$taskProcess.Id;captured_handle_nonzero=($handle -ne [IntPtr]::Zero);clearance_sha256=$actual;exact_arguments=$arguments;automatic_retry=$false;sole_dispatch_owner='root'}
$stream=[System.IO.File]::Open((Join-Path $folder 'dispatch.json'),[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write,[System.IO.FileShare]::Read)
try {$bytes=[System.Text.Encoding]::UTF8.GetBytes(($record | ConvertTo-Json -Depth 5));$stream.Write($bytes,0,$bytes.Length)} finally {$stream.Dispose()}
if($handle -eq [IntPtr]::Zero){throw 'Canonical wrapper handle unavailable; preserve existing attempt.'}
$record | ConvertTo-Json -Compress
