param([Parameter(Mandatory=$true)][string]$ClearanceSha256)
$ErrorActionPreference='Continue'
$runRoot='E:\codex-artifacts\sonic23_teleop_resume_20260911\fast_controller_phase_fit_v1'
foreach($existing in @('fit_process_status.json','fit_stdout.log','fit_stderr.log','fit')){
    if(Test-Path -LiteralPath (Join-Path $runRoot $existing)){throw 'Existing phase fit attempt must remain preserved.'}
}
function Read-Sha256([string]$path){
    $digest=[Security.Cryptography.SHA256]::Create()
    try{return ([BitConverter]::ToString($digest.ComputeHash([IO.File]::ReadAllBytes($path)))).Replace('-','').ToLowerInvariant()}finally{$digest.Dispose()}
}
$clearancePath=Join-Path $runRoot 'training_clearance.json'
if((Read-Sha256 $clearancePath) -ne $ClearanceSha256){throw 'Phase fit clearance identity mismatch.'}
$clearance=Get-Content -LiteralPath $clearancePath -Raw | ConvertFrom-Json
if($clearance.model_fitting_authorized -ne $true -or $clearance.additional_updates -ne 5000 -or $clearance.ordinary_final_global_step -ne 65000){throw 'Phase fit clearance scope mismatch.'}
foreach($required in @('frozen_inputs_v2.json','run_fit_durable.ps1')){
    $requiredPath=[IO.Path]::GetFullPath((Join-Path $runRoot $required))
    $bindings=@($clearance.bound_files | Where-Object {[IO.Path]::GetFullPath($_.path) -ieq $requiredPath})
    if($bindings.Count -ne 1){throw ('Required reviewed input not bound exactly once: '+$required)}
}
foreach($item in $clearance.bound_files){if((Read-Sha256 $item.path) -ne $item.sha256){throw ('Reviewed phase fit input changed: '+$item.path)}}
$env:OMP_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:MKL_NUM_THREADS='1'
$statePath=Join-Path $runRoot 'fit_process_status.json'
$status=[ordered]@{kind='one_phase_fullbatch5000_fit';state='RUNNING';powershell_pid=$PID;started_utc=[DateTime]::UtcNow.ToString('o');additional_updates=5000;ordinary_final_global_step=65000;full_batch_rows=3057;clearance_sha256=$ClearanceSha256;physical_evaluations=0;hardware_authorized=$false}
$status | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $statePath -Encoding UTF8
$runExit=-1;$runError=$null
try{
    & 'C:\Users\camer\AppData\Local\Programs\Python\Python310\python.exe' -u (Join-Path $runRoot 'source_snapshot_v1\fit_phase_fullbatch_once.py') 1> (Join-Path $runRoot 'fit_stdout.log') 2> (Join-Path $runRoot 'fit_stderr.log')
    $runExit=$LASTEXITCODE
}catch{$runError=$_.Exception.Message}finally{
    $status.state='EXITED';$status['ended_utc']=[DateTime]::UtcNow.ToString('o');$status['process_exit_code']=$runExit;$status['launcher_error']=$runError
    $status['verdict']='Read ordinary-final fit/export reports. Canonical physics requires separate final export review and authorization.'
    $status | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $statePath -Encoding UTF8
}
exit $runExit
