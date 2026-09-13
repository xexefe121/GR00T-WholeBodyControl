param([Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$LaunchReceiptSha256)
$ErrorActionPreference='Stop'
$runRoot=$PSScriptRoot
$processRoot=Join-Path $runRoot 'fit_process'
$shareMode=[IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete

function Read-SharedText([string]$Path){
    $stream=[IO.File]::Open($Path,[IO.FileMode]::Open,[IO.FileAccess]::Read,$shareMode)
    try {
        $reader=[IO.StreamReader]::new($stream,[Text.Encoding]::UTF8,$true)
        try { return $reader.ReadToEnd() } finally { $reader.Dispose() }
    } finally { $stream.Dispose() }
}
function Read-SharedJson([string]$Path){ return (Read-SharedText $Path | ConvertFrom-Json) }
function Read-SharedSha([string]$Path){
    $stream=[IO.File]::Open($Path,[IO.FileMode]::Open,[IO.FileAccess]::Read,$shareMode)
    $hasher=[Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($hasher.ComputeHash($stream))).Replace('-','').ToLowerInvariant() }
    finally { $hasher.Dispose();$stream.Dispose() }
}
function Write-NewJson([string]$Path,$Value){
    $stream=[IO.File]::Open($Path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,$shareMode)
    try {
        $writer=[IO.StreamWriter]::new($stream,[Text.UTF8Encoding]::new($false))
        try { $writer.Write(($Value | ConvertTo-Json -Depth 100));$writer.WriteLine();$writer.Flush();$stream.Flush($true) }
        finally { $writer.Dispose() }
    } finally { $stream.Dispose() }
}
function Same-Path([string]$A,[string]$B){
    return [string]::Equals([IO.Path]::GetFullPath($A),[IO.Path]::GetFullPath($B),[StringComparison]::OrdinalIgnoreCase)
}
function Check-Pins($Pins){
    $results=[ordered]@{}
    $all=$true
    foreach($property in $Pins.PSObject.Properties){
        try {
            $actual=Read-SharedSha $property.Name
            $matched=$actual -eq $property.Value
            $results[$property.Name]=[ordered]@{expected=$property.Value;actual=$actual;matched=$matched}
            if(-not $matched){$all=$false}
        } catch {
            $results[$property.Name]=[ordered]@{expected=$property.Value;actual=$null;matched=$false;error=$_.Exception.Message}
            $all=$false
        }
    }
    return [ordered]@{all_exact=$all;count=$results.Count;files=$results}
}

$launch=$null;$lockStream=$null;$child=$null;$rawExit=$null;$runExit=1;$errorText=$null
$started=$false;$prePins=$null;$postPins=$null
foreach($name in @('fit','fit_preflight_failure.json','freeze_failure.json','finalize_failure.json')){
    if(Test-Path -LiteralPath (Join-Path $runRoot $name)){throw "Existing fit attempt must remain preserved: $name"}
}
foreach($name in @('running.lock','start.json','child.json','exit.json','stdout.log','stderr.log','prerun_pins.json','postrun_pins.json')){
    if(Test-Path -LiteralPath (Join-Path $processRoot $name)){throw "Existing process attempt must remain preserved: $name"}
}
[IO.Directory]::CreateDirectory($processRoot) | Out-Null
# CreateNew wins ownership atomically; the retained lock file blocks every rerun.
$lockStream=[IO.File]::Open((Join-Path $processRoot 'running.lock'),[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
try {
    $receiptPath=Join-Path $runRoot 'fit_launch_receipt.json'
    if((Read-SharedSha $receiptPath) -ne $LaunchReceiptSha256){throw 'Launch receipt identity mismatch.'}
    $launch=Read-SharedJson $receiptPath
    if($launch.kind -ne 'one_selected_physical_fit_launch' -or $launch.requested_updates -ne 5000 -or $launch.ordinary_final_step -ne 75000 -or $launch.automatic_resume -ne $false){throw 'Selected fit launch scope mismatch.'}
    if($launch.budgets.valid_rows -lt 1 -or $launch.budgets.valid_rows -gt 3054){throw 'Invalid fixed physical row count.'}
    $valid=[int]$launch.budgets.valid_rows
    $expectedCalls=1126+2*[int][Math]::Ceiling($valid/256.0)
    $expectedRows=5000*(4209+$valid)
    if($launch.budgets.head_onnx_calls -ne $expectedCalls -or $launch.budgets.training_head_rows -ne $expectedRows){throw 'Fixed fit budget mismatch.'}
    $prePins=Check-Pins $launch.input_sha256
    Write-NewJson (Join-Path $processRoot 'prerun_pins.json') $prePins
    if(-not $prePins.all_exact){throw 'Frozen prelaunch inputs changed.'}
    $clearancePath=Join-Path $runRoot 'training_clearance.json'
    if((Read-SharedSha $clearancePath) -ne $launch.clearance_sha256){throw 'Clearance identity mismatch.'}
    $clear=Read-SharedJson $clearancePath
    if($clear.approved -ne $true -or $clear.model_fitting_authorized -ne $true -or $clear.additional_updates -ne 5000 -or $clear.ordinary_final_step -ne 75000 -or $clear.automatic_resume -ne $false){throw 'Fit clearance scope mismatch.'}
    if((Read-SharedSha $clear.final_launch_review_path) -ne $launch.final_review_sha256){throw 'Final launch review changed.'}
    if((Read-SharedSha (Join-Path $runRoot 'run_fit_durable_v1.ps1')) -ne $clear.launcher_sha256){throw 'Frozen launcher changed.'}
    $snapshot=Join-Path $runRoot 'source_snapshot_v1'
    $driver=Join-Path $snapshot 'fit_physical_continuation.py'
    $runtimePython='C:\Users\camer\AppData\Local\Programs\Python\Python310\python.exe'
    if(-not (Same-Path $launch.command.python $runtimePython) -or -not (Same-Path $launch.command.driver $driver) -or -not (Same-Path $launch.command.working_directory $snapshot)){throw 'Frozen absolute command mismatch.'}
    if($launch.command.arguments.Count -ne 2 -or $launch.command.arguments[0] -ne '-u' -or -not (Same-Path $launch.command.arguments[1] $driver)){throw 'Unexpected Python command arguments.'}
    $environment=$launch.command.environment
    if((($environment.PSObject.Properties.Name | Sort-Object) -join ',') -ne 'MKL_NUM_THREADS,OMP_NUM_THREADS,OPENBLAS_NUM_THREADS,PYTHONPATH'){throw 'Unexpected launch environment fields.'}
    foreach($name in @('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')){
        if($environment.$name -ne '1'){throw "Thread setting mismatch: $name"}
        [Environment]::SetEnvironmentVariable($name,'1','Process')
    }
    if(-not (Same-Path $environment.PYTHONPATH $snapshot)){throw 'Frozen import path mismatch.'}
    [Environment]::SetEnvironmentVariable('PYTHONPATH',$snapshot,'Process')
    Write-NewJson (Join-Path $processRoot 'start.json') ([ordered]@{
        kind='one_fixed_physical_fit';wrapper_pid=$PID;started_utc=[DateTime]::UtcNow.ToString('o');
        launch_receipt_sha256=$LaunchReceiptSha256;command=$launch.command;requested_updates=5000;
        ordinary_final_step=75000;budgets=$launch.budgets;automatic_resume=$false
    })
    $argumentLine='-u "'+$driver+'"'
    $child=Start-Process -FilePath $runtimePython -ArgumentList $argumentLine -WorkingDirectory $snapshot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $processRoot 'stdout.log') -RedirectStandardError (Join-Path $processRoot 'stderr.log')
    $started=$true
    # Capture the process handle before waiting; do not Refresh before ExitCode.
    $capturedHandle=$child.Handle
    Write-NewJson (Join-Path $processRoot 'child.json') ([ordered]@{
        wrapper_pid=$PID;child_pid=$child.Id;captured_handle_nonzero=($capturedHandle -ne [IntPtr]::Zero);
        started_utc=[DateTime]::UtcNow.ToString('o');launch_receipt_sha256=$LaunchReceiptSha256
    })
    $child.WaitForExit()
    $rawExit=$child.ExitCode
    if($null -eq $rawExit){throw 'Child exit is unknown; no automatic rerun or success claim.'}
    if($rawExit -ne 0){throw "Python fit exited $rawExit; original outputs remain preserved."}
    $report=Read-SharedJson (Join-Path $runRoot 'fit/report.json')
    foreach($field in @('completed','optimization_completed','final_export_diagnostics_completed','numerical_gate_passed','export_parity_passed')){
        if($report.$field -ne $true){throw "Incomplete fit report: $field"}
    }
    if($report.ordinary_final_step -ne 75000 -or $report.additional_updates -ne 5000 -or $report.head_ONNX_calls -ne $expectedCalls){throw 'Ordinary final report count mismatch.'}
    if((Read-SharedSha (Join-Path $runRoot 'fit/student_head.pt')) -ne $report.checkpoint_sha256 -or (Read-SharedSha (Join-Path $runRoot 'fit/student_head.onnx')) -ne $report.onnx_sha256){throw 'Final output identity mismatch.'}
    $runExit=0
} catch {
    $errorText=$_.Exception.Message
    $runExit=1
} finally {
    if($null -ne $launch){
        $postPins=Check-Pins $launch.input_sha256
        try { Write-NewJson (Join-Path $processRoot 'postrun_pins.json') $postPins } catch { $errorText=($_.Exception.Message);$runExit=1 }
        if(-not $postPins.all_exact){$runExit=1;if($null -eq $errorText){$errorText='Frozen postrun inputs changed.'}}
        try { if((Read-SharedSha (Join-Path $runRoot 'fit_launch_receipt.json')) -ne $LaunchReceiptSha256){throw 'Launch receipt changed.'} }
        catch { $errorText=$_.Exception.Message;$runExit=1 }
    }
    $result=[ordered]@{
        kind='one_fixed_physical_fit_exit';ended_utc=[DateTime]::UtcNow.ToString('o');wrapper_pid=$PID;
        child_pid=$(if($null -ne $child){$child.Id}else{$null});child_started=$started;
        raw_python_exit_code=$rawExit;exit_code=$runExit;exit_known=($null -ne $rawExit);error=$errorText;
        launch_receipt_sha256=$LaunchReceiptSha256;requested_updates=5000;ordinary_final_step=75000;
        all_postrun_pins_exact=$(if($null -ne $postPins){$postPins.all_exact}else{$false});
        automatic_resume=$false;behavioral_qualification=$false
    }
    try { Write-NewJson (Join-Path $processRoot 'exit.json') $result } finally {
        if($null -ne $child){$child.Dispose()}
        if($null -ne $lockStream){$lockStream.Dispose()}
    }
}
exit $runExit
