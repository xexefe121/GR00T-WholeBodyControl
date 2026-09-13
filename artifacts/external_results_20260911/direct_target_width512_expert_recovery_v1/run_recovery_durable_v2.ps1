param([Parameter(Mandatory=$true)][string]$ClearanceSha256)
$ErrorActionPreference='Stop'
$runRoot=$PSScriptRoot
$processRoot=Join-Path $runRoot 'recovery_process_v2'
$share=[IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete
function Get-TaskSha([string]$Path){$s=[IO.File]::Open($Path,[IO.FileMode]::Open,[IO.FileAccess]::Read,$share);$h=[Security.Cryptography.SHA256]::Create();try{return ([BitConverter]::ToString($h.ComputeHash($s))).Replace('-','').ToLowerInvariant()}finally{$h.Dispose();$s.Dispose()}}
function Read-TaskJson([string]$Path){$s=[IO.File]::Open($Path,[IO.FileMode]::Open,[IO.FileAccess]::Read,$share);try{$r=[IO.StreamReader]::new($s);try{return ($r.ReadToEnd()|ConvertFrom-Json)}finally{$r.Dispose()}}finally{$s.Dispose()}}
function Write-TaskJson([string]$Path,$Value){$s=[IO.File]::Open($Path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,$share);try{$w=[IO.StreamWriter]::new($s,[Text.UTF8Encoding]::new($false));try{$w.Write(($Value|ConvertTo-Json -Depth 35));$w.Flush();$s.Flush($true)}finally{$w.Dispose()}}finally{$s.Dispose()}}
function Check-Pins($Pins){$entries=[ordered]@{};$all=$true;foreach($p in $Pins.GetEnumerator()){try{$actual=Get-TaskSha $p.Key;$ok=$actual -eq $p.Value}catch{$actual=$null;$ok=$false};$entries[$p.Key]=[ordered]@{expected=$p.Value;actual=$actual;matched=$ok};if(-not $ok){$all=$false}};return [ordered]@{all_exact=$all;files=$entries}}
$clearancePath=Join-Path $runRoot 'execution_clearance_v2.json'
if((Get-TaskSha $clearancePath) -ne $ClearanceSha256){throw 'Recovery clearance hash changed.'}
$clearance=Read-TaskJson $clearancePath
$receiptPath=Join-Path $runRoot 'launch_receipt_v2.json'
$requestPath=Join-Path $runRoot 'execution_request.json'
$frozenPath=Join-Path $runRoot 'frozen_inputs.json'
if($clearance.approved -ne $true -or $clearance.launch_receipt_sha256 -ne (Get-TaskSha $receiptPath) -or $clearance.request_sha256 -ne (Get-TaskSha $requestPath) -or $clearance.frozen_receipt_sha256 -ne (Get-TaskSha $frozenPath)){throw 'Recovery exact selection is absent.'}
$request=Read-TaskJson $requestPath
if($request.root_selected -ne $true -or $request.protocol.initial_global_control -ne 251 -or $request.protocol.actual_native_step_max -ne 15680){throw 'Wrong fixed recovery protocol.'}
if((Get-TaskSha $clearance.review.path) -ne $clearance.review.sha256){throw 'Root concrete review changed.'}
$review=Read-TaskJson $clearance.review.path
if($review.($clearance.review.pass_field) -ne $true -or $review.request_sha256 -ne $clearance.request_sha256 -or $review.frozen_receipt_sha256 -ne $clearance.frozen_receipt_sha256 -or $review.launch_receipt_sha256 -ne $clearance.launch_receipt_sha256){throw 'Root review does not bind actual recovery.'}
$receipt=Read-TaskJson $receiptPath
if($receipt.request_sha256 -ne $clearance.request_sha256 -or $receipt.frozen_receipt_sha256 -ne $clearance.frozen_receipt_sha256){throw 'Launch receipt describes another request.'}
$pins=[ordered]@{}
foreach($p in $receipt.input_sha256.PSObject.Properties){if($pins.Contains($p.Name)){throw 'Duplicate launch pin.'};$pins[$p.Name]=$p.Value}
foreach($p in @(@($receiptPath,$clearance.launch_receipt_sha256),@($clearancePath,$ClearanceSha256),@($clearance.review.path,$clearance.review.sha256))){if($pins.Contains($p[0]) -and $pins[$p[0]] -ne $p[1]){throw 'Conflicting root launch pin.'};$pins[$p[0]]=$p[1]}
foreach($name in @('ATTEMPT_STARTED','initial_seed','nominal','outcome.json','failure.json')){if(Test-Path -LiteralPath (Join-Path $runRoot $name)){throw 'Previous recovery attempt must remain preserved.'}}
[IO.Directory]::CreateDirectory($processRoot)|Out-Null
$lock=[IO.File]::Open((Join-Path $processRoot 'started.lock'),[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
$raw=$null;$runExit=1;$errorText=$null;$child=$null;$post=$null;$countersFinal=$false;$branchComplete=$false
try{
    $pre=Check-Pins $pins;Write-TaskJson (Join-Path $processRoot 'prerun_pins.json') $pre
    if(-not $pre.all_exact){throw 'Recovery inputs changed before execution.'}
    Write-TaskJson (Join-Path $processRoot 'start.json') ([ordered]@{wrapper_pid=$PID;started_utc=[DateTime]::UtcNow.ToString('o');request_sha256=$clearance.request_sha256;frozen_receipt_sha256=$clearance.frozen_receipt_sha256;clearance_sha256=$ClearanceSha256;launch_receipt_sha256=$clearance.launch_receipt_sha256;arguments=$receipt.wsl_arguments;initial_global_control=251;requested_native_steps=15680})
    $arguments=($receipt.wsl_arguments | ForEach-Object {if(($_ -match '\s') -or $_.IndexOf([char]34) -ge 0 -or $_.IndexOf([char]39) -ge 0){throw 'Invalid fixed WSL argument.'};$_}) -join ' '
    $child=Start-Process -FilePath "$env:SystemRoot\System32\wsl.exe" -ArgumentList $arguments -WorkingDirectory $runRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $processRoot 'stdout.log') -RedirectStandardError (Join-Path $processRoot 'stderr.log')
    $handle=$child.Handle
    Write-TaskJson (Join-Path $processRoot 'child.json') ([ordered]@{wrapper_pid=$PID;child_pid=$child.Id;handle_acquired=($handle -ne [IntPtr]::Zero);command_arguments=$arguments})
    $child.WaitForExit();$raw=$child.ExitCode
    if($null -eq $raw){throw 'Unknown WSL/Python recovery exit.'}
    if(Test-Path -LiteralPath (Join-Path $runRoot 'work_counters.json')){$counts=Read-TaskJson (Join-Path $runRoot 'work_counters.json');$countersFinal=($counts.phase -eq 'task_ended' -and @($counts.active).Count -eq 0)}
    if(Test-Path -LiteralPath (Join-Path $runRoot 'driver_completion.json')){$completion=Read-TaskJson (Join-Path $runRoot 'driver_completion.json');$branchComplete=($completion.requested_branch_completed -eq $true)}
    if($raw -ne 0){throw "Recovery ended with raw exit $raw; no retry."}
    if(-not $countersFinal -or -not $branchComplete){throw 'Complete branch and final work accounting not present.'}
    $runExit=0
}catch{$errorText=$_.Exception.Message;$runExit=1}
finally{
    $post=Check-Pins $pins;Write-TaskJson (Join-Path $processRoot 'postrun_pins.json') $post
    if(-not $post.all_exact){$runExit=1;$errorText='Postrun recovery pins changed.'}
    Write-TaskJson (Join-Path $processRoot 'exit.json') ([ordered]@{wrapper_pid=$PID;child_pid=$(if($null -ne $child){$child.Id}else{$null});ended_utc=[DateTime]::UtcNow.ToString('o');raw_python_exit_code=$raw;raw_exit_known=($null -ne $raw);exit_code=$runExit;error=$errorText;all_postrun_pins_exact=$post.all_exact;final_work_counters_present=$countersFinal;requested_branch_completed=$branchComplete;physical_and_intent_qualification_pending=$true;automatic_retry=$false})
    if($null -ne $child){$child.Dispose()};$lock.Dispose()
}
exit $runExit
