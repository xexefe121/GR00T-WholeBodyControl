param([Parameter(Mandatory=$true)][string]$ClearanceSha256)
$ErrorActionPreference='Continue'
$runRoot='E:\codex-artifacts\sonic23_teleop_resume_20260911\velocity_chord_student_v1'
foreach($existing in @('fit_process_status.json','fit_stdout.log','fit_stderr.log','fit','fit_preflight_failure.json')){
    if(Test-Path -LiteralPath (Join-Path $runRoot $existing)){throw 'Existing velocity-chord fit attempt must remain preserved.'}
}
function Read-Sha256([string]$path){
    $digest=[Security.Cryptography.SHA256]::Create()
    try{return ([BitConverter]::ToString($digest.ComputeHash([IO.File]::ReadAllBytes($path)))).Replace('-','').ToLowerInvariant()}finally{$digest.Dispose()}
}
$clearancePath=Join-Path $runRoot 'training_clearance.json'
if((Read-Sha256 $clearancePath) -ne $ClearanceSha256){throw 'Training clearance identity mismatch.'}
$clearance=Get-Content -LiteralPath $clearancePath -Raw | ConvertFrom-Json
if($clearance.approved -ne $true -or $clearance.additional_updates -ne 5000 -or $clearance.ordinary_final_step -ne 70000){throw 'Velocity-chord fit clearance scope mismatch.'}
if((Read-Sha256 (Join-Path $runRoot 'training_frozen_inputs.json')) -ne $clearance.frozen_receipt_sha256){throw 'Training source receipt binding mismatch.'}
if((Read-Sha256 (Join-Path $runRoot 'run_fit_durable.ps1')) -ne $clearance.launcher_sha256){throw 'Training launcher binding mismatch.'}
if((Read-Sha256 $clearance.source_review_path) -ne $clearance.source_review_sha256){throw 'Training source review binding mismatch.'}
if((Read-Sha256 $clearance.dataset_review_path) -ne $clearance.dataset_review_sha256){throw 'Generated data review binding mismatch.'}
$env:OMP_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:MKL_NUM_THREADS='1'
$status=[ordered]@{kind='one_fixed5000_velocity_chord_fit';state='RUNNING';powershell_pid=$PID;started_utc=[DateTime]::UtcNow.ToString('o');additional_updates=5000;ordinary_final_step=70000;nominal_rows_per_update=3057;axis_pairs_per_update=576;maximum_head_ONNX_calls=1126;clearance_sha256=$ClearanceSha256;physical_evaluations=0;hardware_authorized=$false}
$statePath=Join-Path $runRoot 'fit_process_status.json'
$status | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $statePath -Encoding UTF8
$rawExit=-1;$runExit=-1;$runError=$null
try{
    & 'C:\Users\camer\AppData\Local\Programs\Python\Python310\python.exe' -u (Join-Path $runRoot 'source_snapshot_v3\fit_velocity_chords.py') 1> (Join-Path $runRoot 'fit_stdout.log') 2> (Join-Path $runRoot 'fit_stderr.log')
    $rawExit=$LASTEXITCODE;$runExit=$rawExit
    if($rawExit -eq 0){
        $reportPath=Join-Path $runRoot 'fit/report.json'
        if(-not(Test-Path -LiteralPath $reportPath)){$runExit=5}else{
            $report=Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
            if($report.completed -ne $true -or $report.numerical_gate_passed -ne $true -or $report.ordinary_final_step -ne 70000 -or $report.head_ONNX_calls -ne 1126){$runExit=5}
        }
    }
}catch{$runError=$_.Exception.Message;$runExit=7}finally{
    $status.state='EXITED';$status['ended_utc']=[DateTime]::UtcNow.ToString('o');$status['raw_python_exit_code']=$rawExit;$status['process_exit_code']=$runExit;$status['launcher_error']=$runError
    $status['verdict']='Ordinary-final optimization and numerical artifacts preserved. Canonical rollout requires separate final export review.'
    $status | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $statePath -Encoding UTF8
}
exit $runExit
