param([Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$ClearanceSha256,
      [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$LaunchReceiptSha256)
$ErrorActionPreference='Stop'
$base=$PSScriptRoot
$processRoot=Join-Path $base 'fit_process_v1'
if(Test-Path -LiteralPath $processRoot){throw 'Existing dispatch/attempt preserved; no retry.'}
if(Test-Path -LiteralPath (Join-Path $base 'fit')){throw 'Existing fit preserved.'}
function Sha([string]$Path){
    $s=[IO.File]::OpenRead($Path);$h=[Security.Cryptography.SHA256]::Create()
    try{return ([BitConverter]::ToString($h.ComputeHash($s))).Replace('-','').ToLowerInvariant()}finally{$h.Dispose();$s.Dispose()}
}
function Write-New([string]$Path,$Value){
    $s=[IO.File]::Open($Path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
    try{$w=[IO.StreamWriter]::new($s,[Text.UTF8Encoding]::new($false));try{$w.Write(($Value|ConvertTo-Json -Depth 20));$w.WriteLine();$w.Flush();$s.Flush($true)}finally{$w.Dispose()}}finally{$s.Dispose()}
}
if((Sha (Join-Path $base 'training_clearance.json')) -ne $ClearanceSha256 -or (Sha (Join-Path $base 'launch_receipt.json')) -ne $LaunchReceiptSha256){throw 'Exact selected dispatch identities required.'}
$clear=Get-Content -LiteralPath (Join-Path $base 'training_clearance.json') -Raw|ConvertFrom-Json
if($clear.approved -ne $true -or $clear.launch_receipt_sha256 -ne $LaunchReceiptSha256){throw 'Concrete clearance absent.'}
$exe='C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe'
$launcher=Join-Path $base 'run_fit_durable_v1.ps1'
if((Sha $launcher) -ne $clear.launcher_sha256){throw 'Launcher changed.'}
[IO.Directory]::CreateDirectory($processRoot)|Out-Null
$argsLiteral=@('-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',$launcher,'-ClearanceSha256',$ClearanceSha256,'-LaunchReceiptSha256',$LaunchReceiptSha256)
Write-New (Join-Path $processRoot 'dispatch_attempt.json') ([ordered]@{started_utc=[DateTime]::UtcNow.ToString('o');dispatcher_pid=$PID;executable=$exe;arguments=$argsLiteral;clearance_sha256=$ClearanceSha256;launch_receipt_sha256=$LaunchReceiptSha256;automatic_retry=$false})
$wrapper=$null
try{
    $command='-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "'+$launcher+'" -ClearanceSha256 '+$ClearanceSha256+' -LaunchReceiptSha256 '+$LaunchReceiptSha256
    $wrapper=Start-Process -FilePath $exe -ArgumentList $command -WorkingDirectory $base -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $processRoot 'wrapper_stdout.log') -RedirectStandardError (Join-Path $processRoot 'wrapper_stderr.log')
    $handle=$wrapper.Handle
    Write-New (Join-Path $processRoot 'dispatch.json') ([ordered]@{started_utc=[DateTime]::UtcNow.ToString('o');dispatcher_pid=$PID;wrapper_pid=$wrapper.Id;captured_handle_nonzero=($handle -ne [IntPtr]::Zero);executable=$exe;arguments=$argsLiteral;clearance_sha256=$ClearanceSha256;launch_receipt_sha256=$LaunchReceiptSha256;automatic_retry=$false})
    [ordered]@{wrapper_pid=$wrapper.Id;captured_handle_nonzero=($handle -ne [IntPtr]::Zero);launch_receipt_sha256=$LaunchReceiptSha256}|ConvertTo-Json
}catch{
    Write-New (Join-Path $processRoot 'dispatch_failure.json') ([ordered]@{error=$_.Exception.Message;wrapper_pid=$(if($null -eq $wrapper){$null}else{$wrapper.Id});automatic_retry=$false})
    throw
}finally{if($null -ne $wrapper){$wrapper.Dispose()}}
