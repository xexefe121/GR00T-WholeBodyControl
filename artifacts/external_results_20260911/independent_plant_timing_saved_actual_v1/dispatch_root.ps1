$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
foreach ($name in @('dispatch.json','dispatch.lock','process_v1','results_v1')) {
    if (Test-Path -LiteralPath (Join-Path $taskRoot $name)) { throw "Preserve existing attempt: $name" }
}
$taskReceiptSha = '4559e431a0ca69fc714e08fc583098f030faf8097306d1b865ae1463fdf09374'
$taskClearance = Get-Content -Raw -LiteralPath (Join-Path $taskRoot 'launch_clearance.json') | ConvertFrom-Json
if ($taskClearance.root_selected_single_saved_audit -ne $true -or $taskClearance.launch_receipt_sha256 -ne $taskReceiptSha) { throw 'Actual clearance required.' }
$taskLock = [IO.File]::Open((Join-Path $taskRoot 'dispatch.lock'),[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
try {
    $taskArguments = @('-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',(Join-Path $taskRoot 'run_audit_durable.ps1'),'-LaunchReceiptSha256',$taskReceiptSha)
    $taskProcess = Start-Process -FilePath "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -ArgumentList $taskArguments -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskRoot 'dispatch.stdout.log') -RedirectStandardError (Join-Path $taskRoot 'dispatch.stderr.log')
    $taskHandle = $taskProcess.Handle
    $taskRecord = [ordered]@{wrapper_pid=$taskProcess.Id;handle_acquired=($taskHandle -ne [IntPtr]::Zero);utc=[DateTime]::UtcNow.ToString('o');launch_receipt_sha256=$taskReceiptSha;arguments=$taskArguments;automatic_retry=$false}
    $taskBytes = [Text.UTF8Encoding]::new($false).GetBytes(($taskRecord | ConvertTo-Json -Depth 8))
    $taskStream = [IO.File]::Open((Join-Path $taskRoot 'dispatch.json'),[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
    try { $taskStream.Write($taskBytes,0,$taskBytes.Length) } finally { $taskStream.Dispose() }
    $taskRecord | ConvertTo-Json -Depth 8
    $taskProcess.Dispose()
} finally { $taskLock.Dispose() }
