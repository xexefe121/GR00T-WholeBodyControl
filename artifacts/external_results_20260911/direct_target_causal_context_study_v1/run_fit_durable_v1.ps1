param([Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$ClearanceSha256)
$ErrorActionPreference='Stop'
$runRoot=$PSScriptRoot
$processRoot=Join-Path $runRoot 'fit_process'
$shareMode=[IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete
function Read-SharedText([string]$Path){
    $stream=[IO.File]::Open($Path,[IO.FileMode]::Open,[IO.FileAccess]::Read,$shareMode)
    try {$reader=[IO.StreamReader]::new($stream,[Text.Encoding]::UTF8,$true);try{return $reader.ReadToEnd()}finally{$reader.Dispose()}}finally{$stream.Dispose()}
}
function Read-SharedJson([string]$Path){return (Read-SharedText $Path | ConvertFrom-Json)}
function Read-SharedSha([string]$Path){
    $stream=[IO.File]::Open($Path,[IO.FileMode]::Open,[IO.FileAccess]::Read,$shareMode);$hasher=[Security.Cryptography.SHA256]::Create()
    try{return ([BitConverter]::ToString($hasher.ComputeHash($stream))).Replace('-','').ToLowerInvariant()}finally{$hasher.Dispose();$stream.Dispose()}
}
function Write-NewJson([string]$Path,$Value){
    $stream=[IO.File]::Open($Path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,$shareMode)
    try{$writer=[IO.StreamWriter]::new($stream,[Text.UTF8Encoding]::new($false));try{$writer.Write(($Value|ConvertTo-Json -Depth 100));$writer.WriteLine();$writer.Flush();$stream.Flush($true)}finally{$writer.Dispose()}}finally{$stream.Dispose()}
}
function Check-Pins($Pins){
    $results=[ordered]@{};$all=$true
    foreach($property in $Pins.PSObject.Properties){
        try{$actual=Read-SharedSha $property.Name;$matched=$actual -eq $property.Value;$results[$property.Name]=[ordered]@{expected=$property.Value;actual=$actual;matched=$matched};if(-not $matched){$all=$false}}
        catch{$results[$property.Name]=[ordered]@{expected=$property.Value;actual=$null;matched=$false;error=$_.Exception.Message};$all=$false}
    }
    return [ordered]@{all_exact=$all;count=$results.Count;files=$results}
}
$child=$null;$rawExit=$null;$runExit=1;$errorText=$null;$started=$false;$pins=$null;$postPins=$null
if(Test-Path -LiteralPath (Join-Path $runRoot 'fit')){throw 'Existing fit must remain preserved; no automatic rerun.'}
foreach($name in @('running.lock','start.json','child.json','exit.json','stdout.log','stderr.log','prerun_pins.json','postrun_pins.json')){
    if(Test-Path -LiteralPath (Join-Path $processRoot $name)){throw "Existing process attempt must remain preserved: $name"}
}
[IO.Directory]::CreateDirectory($processRoot)|Out-Null
$lockStream=[IO.File]::Open((Join-Path $processRoot 'running.lock'),[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
try{
    $clearancePath=Join-Path $runRoot 'training_clearance.json'
    if((Read-SharedSha $clearancePath) -ne $ClearanceSha256){throw 'Clearance identity differs.'}
    $clear=Read-SharedJson $clearancePath
    if($clear.approved -ne $true -or $clear.updates_per_condition -ne 3000 -or $clear.ordinary_final_step -ne 68000 -or (@($clear.conditions) -join ',') -ne 'blinded,causal' -or $clear.automatic_retry -ne $false){throw 'Selected direct fit scope differs.'}
    $receiptPath=Join-Path $runRoot 'training_frozen_inputs.json';$requestPath=Join-Path $runRoot 'training_request.json'
    if((Read-SharedSha $receiptPath) -ne $clear.frozen_receipt_sha256 -or (Read-SharedSha $requestPath) -ne $clear.request_sha256){throw 'Frozen request or receipt differs.'}
    $receipt=Read-SharedJson $receiptPath;$request=Read-SharedJson $requestPath
    if($request.kind -ne 'matched_causal_context_utility_study' -or $request.root_selected -ne $true -or $request.updates_per_condition -ne 3000 -or (@($request.conditions) -join ',') -ne 'blinded,causal' -or $request.coefficient -ne 1.8188207859141674){throw 'Unexpected request.'}
    if((Read-SharedSha $clear.review_path) -ne $clear.review_sha256){throw 'Review identity differs.'}
    $review=Read-SharedJson $clear.review_path
    if($review.($clear.review_pass_field) -ne $true -or $review.training_request_sha256 -ne $clear.request_sha256 -or $review.frozen_receipt_sha256 -ne $clear.frozen_receipt_sha256){throw 'Review does not clear this exact request and receipt.'}
    if((Read-SharedSha $PSCommandPath) -ne $clear.launcher_sha256){throw 'Launcher identity differs.'}
    $pins=[ordered]@{}
    foreach($property in $receipt.input_sha256.PSObject.Properties){$pins[$property.Name]=$property.Value}
    foreach($property in $receipt.source_sha256.PSObject.Properties){$pins[(Join-Path $receipt.source_directory $property.Name)]=$property.Value}
    $pins[$receiptPath]=$clear.frozen_receipt_sha256;$pins[$requestPath]=$clear.request_sha256
    $pins[$clearancePath]=$ClearanceSha256;$pins[$clear.review_path]=$clear.review_sha256
    $pins=[PSCustomObject]$pins
    $prePins=Check-Pins $pins;Write-NewJson (Join-Path $processRoot 'prerun_pins.json') $prePins
    if(-not $prePins.all_exact){throw 'Prelaunch pins differ.'}
    $snapshot=Join-Path $runRoot 'source_snapshot_v1';$driver=Join-Path $snapshot 'train_context_pair.py'
    $runtimePython='E:\codex_sonic_runtime\direct_target_gpu_20260911\venv\Scripts\python.exe'
    if([IO.Path]::GetFullPath($request.runtime.python_path) -ne [IO.Path]::GetFullPath($runtimePython)){throw 'Wrong isolated Python.'}
    foreach($name in @('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')){[Environment]::SetEnvironmentVariable($name,'1','Process')}
    [Environment]::SetEnvironmentVariable('PYTHONPATH',$snapshot,'Process')
    [Environment]::SetEnvironmentVariable('CUBLAS_WORKSPACE_CONFIG',':4096:8','Process')
    Write-NewJson (Join-Path $processRoot 'start.json') ([ordered]@{
        kind='matched_causal_context_utility_study';wrapper_pid=$PID;started_utc=[DateTime]::UtcNow.ToString('o');
        clearance_sha256=$ClearanceSha256;request_sha256=$clear.request_sha256;frozen_receipt_sha256=$clear.frozen_receipt_sha256;
        command=[ordered]@{python=$runtimePython;arguments=@('-u',$driver);working_directory=$snapshot;CUBLAS_WORKSPACE_CONFIG=':4096:8';threads=1};
        updates_per_condition=3000;ordinary_final_step=68000;conditions=@('blinded','causal');budgets=$request.budgets;automatic_retry=$false
    })
    $child=Start-Process -FilePath $runtimePython -ArgumentList ('-u "'+$driver+'"') -WorkingDirectory $snapshot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $processRoot 'stdout.log') -RedirectStandardError (Join-Path $processRoot 'stderr.log')
    $started=$true;$capturedHandle=$child.Handle
    Write-NewJson (Join-Path $processRoot 'child.json') ([ordered]@{wrapper_pid=$PID;child_pid=$child.Id;captured_handle_nonzero=($capturedHandle -ne [IntPtr]::Zero);started_utc=[DateTime]::UtcNow.ToString('o')})
    $child.WaitForExit();$rawExit=$child.ExitCode
    if($null -eq $rawExit){throw 'Child exit unknown; no success claim or automatic rerun.'}
    if($rawExit -ne 0){throw "Python fit exited $rawExit; all evidence retained."}
    $pair=Read-SharedJson (Join-Path $runRoot 'fit/paired_report.json')
    if($pair.completed -ne $true -or $pair.updates_per_condition -ne 3000 -or $pair.ordinary_final_step -ne 68000 -or (@($pair.conditions) -join ',') -ne 'blinded,causal'){throw 'Fixed pair incomplete.'}
    foreach($field in @('shared_initial_actor_optimizer_RNG_normalization_exact','shared_schedule_exact','all_frozen_inputs_unchanged')){if($pair.$field -ne $true){throw "Pair proof failed: $field"}}
    if($pair.budgets.training_forward_rows -ne 88116000 -or $pair.budgets.training_forward_calls -ne 18000 -or $pair.budgets.training_updates -ne 6000){throw 'Pair budget differs.'}
    foreach($condition in @('blinded','causal')){
        $conditionRoot=Join-Path (Join-Path $runRoot 'fit') $condition
        $reportPath=Join-Path $conditionRoot 'report.json';$report=Read-SharedJson $reportPath
        if((Read-SharedSha $reportPath) -ne $pair.condition_report_sha256.$condition){throw 'Condition report identity differs.'}
        foreach($field in @('completed','optimization_completed','final_export_diagnostics_completed','numerical_gate_passed','export_parity_passed','all_frozen_inputs_unchanged')){if($report.$field -ne $true){throw "Incomplete $condition report: $field"}}
        if($report.condition -ne $condition -or $report.ordinary_final_step -ne 68000 -or $report.additional_updates -ne 3000 -or $report.optimizer_step -ne 3000 -or $report.features -ne 1323 -or $report.head_output -ne 'normalized_target'){throw 'Final condition metadata differs.'}
        if($report.counts.training.rows_verified -ne 44058000 -or $report.counts.training.calls_verified -ne 9000 -or $report.counts.calibration_forward_calls -ne 0 -or $report.counts.calibration_gradient_calls -ne 0){throw 'Condition training counts differ.'}
        foreach($backend in @('initial_GPU32','final_GPU32','CPU64','GPU64','ORT64')){if($report.counts.diagnostics.$backend.rows_verified -ne 367570 -or $report.counts.diagnostics.$backend.calls_verified -ne 1437){throw "Incomplete backend counts: $condition/$backend"}}
        if((Read-SharedSha (Join-Path $conditionRoot 'student_head.pt')) -ne $report.checkpoint_sha256 -or (Read-SharedSha (Join-Path $conditionRoot 'student_head.onnx')) -ne $report.onnx_sha256){throw 'Output identities differ.'}
    }
    $runExit=0
}catch{$errorText=$_.Exception.Message;$runExit=1}
finally{
    if($null -ne $pins){
        $postPins=Check-Pins $pins
        try{Write-NewJson (Join-Path $processRoot 'postrun_pins.json') $postPins}catch{$errorText=$_.Exception.Message;$runExit=1}
        if(-not $postPins.all_exact){$runExit=1;if($null -eq $errorText){$errorText='Postrun pins changed.'}}
    }
    try{Write-NewJson (Join-Path $processRoot 'exit.json') ([ordered]@{
        kind='matched_causal_context_utility_study_exit';ended_utc=[DateTime]::UtcNow.ToString('o');wrapper_pid=$PID;
        child_pid=$(if($null -ne $child){$child.Id}else{$null});child_started=$started;raw_python_exit_code=$rawExit;
        exit_code=$runExit;exit_known=($null -ne $rawExit);error=$errorText;clearance_sha256=$ClearanceSha256;
        all_postrun_pins_exact=$(if($null -ne $postPins){$postPins.all_exact}else{$false});automatic_retry=$false;behavioral_qualification=$false
    })}finally{if($null -ne $child){$child.Dispose()};$lockStream.Dispose()}
}
exit $runExit
