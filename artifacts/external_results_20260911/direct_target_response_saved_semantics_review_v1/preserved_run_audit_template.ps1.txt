param([Parameter(Mandatory=$true)][string]$LaunchReceiptSha256)
$ErrorActionPreference='Stop'
$runRoot=$PSScriptRoot
$processRoot=Join-Path $runRoot 'process_v1'
$share=[IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete
function Get-TaskSha([string]$Path){$s=[IO.File]::Open($Path,[IO.FileMode]::Open,[IO.FileAccess]::Read,$share);$h=[Security.Cryptography.SHA256]::Create();try{return ([BitConverter]::ToString($h.ComputeHash($s))).Replace('-','').ToLowerInvariant()}finally{$h.Dispose();$s.Dispose()}}
function Read-TaskJson([string]$Path){$s=[IO.File]::Open($Path,[IO.FileMode]::Open,[IO.FileAccess]::Read,$share);try{$r=[IO.StreamReader]::new($s);try{return ($r.ReadToEnd()|ConvertFrom-Json)}finally{$r.Dispose()}}finally{$s.Dispose()}}
function Write-TaskJson([string]$Path,$Value){$s=[IO.File]::Open($Path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,$share);try{$w=[IO.StreamWriter]::new($s,[Text.UTF8Encoding]::new($false));try{$w.Write(($Value|ConvertTo-Json -Depth 30));$w.Flush();$s.Flush($true)}finally{$w.Dispose()}}finally{$s.Dispose()}}
function Check-Pins($Pins){$entries=[ordered]@{};$all=$true;foreach($p in $Pins.PSObject.Properties){try{$actual=Get-TaskSha $p.Name;$ok=$actual -eq $p.Value}catch{$actual=$null;$ok=$false};$entries[$p.Name]=[ordered]@{expected=$p.Value;actual=$actual;matched=$ok};if(-not $ok){$all=$false}};return [ordered]@{all_exact=$all;files=$entries}}
$receiptPath=Join-Path $runRoot 'launch_receipt.json'
if((Get-TaskSha $receiptPath) -ne $LaunchReceiptSha256){throw 'Frozen audit launch receipt changed.'}
$receipt=Read-TaskJson $receiptPath
if($receipt.selected_single_saved_audit -ne $true -or $receipt.request_sha256 -ne 'bfb87d2c4dde5e058dd15795c25407d1bd5c914f89af60e62f5499511b4d4898'){throw 'Wrong selected saved audit.'}
if(Test-Path -LiteralPath (Join-Path $runRoot 'results_v1')){throw 'Existing audit results must remain preserved.'}
[IO.Directory]::CreateDirectory($processRoot)|Out-Null
$lock=[IO.File]::Open((Join-Path $processRoot 'started.lock'),[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
$raw=$null;$runExit=1;$errorText=$null;$child=$null;$post=$null
try{
    $pre=Check-Pins $receipt.input_sha256;Write-TaskJson (Join-Path $processRoot 'prerun_pins.json') $pre
    if(-not $pre.all_exact){throw 'Audit launch inputs changed.'}
    foreach($name in @('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')){[Environment]::SetEnvironmentVariable($name,'1','Process')}
    [Environment]::SetEnvironmentVariable('PYTHONDONTWRITEBYTECODE','1','Process')
    $source=$runRoot
    [Environment]::SetEnvironmentVariable('PYTHONPATH',$source,'Process')
    Write-TaskJson (Join-Path $processRoot 'start.json') ([ordered]@{wrapper_pid=$PID;started_utc=[DateTime]::UtcNow.ToString('o');request_sha256=$receipt.request_sha256;launch_receipt_sha256=$LaunchReceiptSha256;source_review_sha256=$receipt.source_review_sha256;task_model_calls=0;native_steps=0})
    $arguments='-d Ubuntu-22.04 --cd / -- bash /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/direct_target_full_state_saved_semantics_review_v1 /mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python -B -u /mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/direct_target_full_state_saved_semantics_review_v1/audit_saved.py'
    $child=Start-Process -FilePath $receipt.wsl_path -ArgumentList $arguments -WorkingDirectory $source -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $processRoot 'stdout.log') -RedirectStandardError (Join-Path $processRoot 'stderr.log')
    $handle=$child.Handle
    Write-TaskJson (Join-Path $processRoot 'child.json') ([ordered]@{wrapper_pid=$PID;child_pid=$child.Id;handle_acquired=($handle -ne [IntPtr]::Zero)})
    $child.WaitForExit();$raw=$child.ExitCode
    if($null -eq $raw){throw 'Unknown Python audit exit.'}
    if($raw -ne 0){throw "Saved audit failed with exit $raw; no retry."}
    $result=Read-TaskJson (Join-Path $runRoot 'results_v1/report.json')
    if($result.evidence_audit_passed -ne $true){throw 'Saved evidence audit did not pass.'}
    $runExit=0
}catch{$errorText=$_.Exception.Message;$runExit=1}
finally{
    $post=Check-Pins $receipt.input_sha256
    Write-TaskJson (Join-Path $processRoot 'postrun_pins.json') $post
    if(-not $post.all_exact){$runExit=1;$errorText='Postrun audit pins changed.'}
    Write-TaskJson (Join-Path $processRoot 'exit.json') ([ordered]@{wrapper_pid=$PID;child_pid=$(if($null -ne $child){$child.Id}else{$null});ended_utc=[DateTime]::UtcNow.ToString('o');raw_python_exit_code=$raw;raw_exit_known=($null -ne $raw);exit_code=$runExit;error=$errorText;all_postrun_pins_exact=$post.all_exact;launch_receipt_sha256=$LaunchReceiptSha256;automatic_retry=$false})
    if($null -ne $child){$child.Dispose()};$lock.Dispose()
}
exit $runExit
