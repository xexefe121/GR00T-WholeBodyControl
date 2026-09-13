$ErrorActionPreference = 'Stop'
$base = 'E:/codex-artifacts/sonic23_teleop_resume_20260911/broader_labels_independent_v1'
$taskChild = Start-Process -FilePath "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',(Join-Path $base 'exit7_stub.ps1')) -WindowStyle Hidden -PassThru
$nativeHandle = $taskChild.Handle
$taskChild.WaitForExit()
$actualExit = $taskChild.ExitCode
if ($null -eq $actualExit -or $actualExit -ne 7) { throw 'Exit-code capture failed.' }
[ordered]@{ expected_exit=7; actual_exit=$actualExit; handle_acquired_before_wait=($nativeHandle -ne [IntPtr]::Zero); no_refresh=$true; utc=[DateTime]::UtcNow.ToString('o'); powershell_version=$PSVersionTable.PSVersion.ToString() } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $base 'exit_capture_test.json') -Encoding UTF8
