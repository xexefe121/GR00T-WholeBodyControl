param([Parameter(Mandatory=$true)][string]$QualificationPath,[Parameter(Mandatory=$true)][string]$QualificationSha256,[Parameter(Mandatory=$true)][string]$FrozenReceiptSha256)
$ErrorActionPreference='Continue'
$runRoot='E:\codex-artifacts\sonic23_teleop_resume_20260911\bfm_entry250_labels_v1'
foreach($existing in @('process_status.json','stdout.log','stderr.log','labels','compatibility')) {
    if(Test-Path -LiteralPath (Join-Path $runRoot $existing)){throw 'Existing collection attempt must remain preserved.'}
}
function Read-Sha256([string]$path){
    $digest=[Security.Cryptography.SHA256]::Create()
    try{return ([BitConverter]::ToString($digest.ComputeHash([IO.File]::ReadAllBytes($path)))).Replace('-','').ToLowerInvariant()}finally{$digest.Dispose()}
}
if((Read-Sha256 $QualificationPath) -ne $QualificationSha256){throw 'Root qualification identity mismatch.'}
if((Read-Sha256 (Join-Path $runRoot 'collector_frozen_inputs.json')) -ne $FrozenReceiptSha256){throw 'Frozen collector identity mismatch.'}
$qualification=Get-Content -LiteralPath $QualificationPath -Raw | ConvertFrom-Json
if($qualification.root_authorized_extraction -ne $true -or $qualification.model_fitting_authorized -ne $false){throw 'Root qualification does not authorize bounded collection.'}
$qualificationFull=[IO.Path]::GetFullPath($QualificationPath).Replace('\','/')
$qualificationWsl='/mnt/'+$qualificationFull.Substring(0,1).ToLowerInvariant()+$qualificationFull.Substring(2)
$statePath=Join-Path $runRoot 'process_status.json'
$status=[ordered]@{kind='one_BFM250_expert_collection_and_saved_compatibility';state='RUNNING_COLLECTION';powershell_pid=$PID;started_utc=[DateTime]::UtcNow.ToString('o');samples=1019;qualification_sha256=$QualificationSha256;frozen_receipt_sha256=$FrozenReceiptSha256;model_fitting_authorized=$false;hardware_authorized=$false}
$status | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $statePath -Encoding UTF8
$runExit=-1;$runError=$null
try{
    & wsl.exe -d Ubuntu-22.04 --cd /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof -- bash /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python -u /mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/bfm_entry250_labels_v1/source_snapshot_v1/collect_bfm250_labels.py --qualification $qualificationWsl 1> (Join-Path $runRoot 'stdout.log') 2> (Join-Path $runRoot 'stderr.log')
    $runExit=$LASTEXITCODE
    if($runExit -eq 0){
        $status.state='RUNNING_SAVED_ARRAY_COMPATIBILITY';$status | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $statePath -Encoding UTF8
        & wsl.exe -d Ubuntu-22.04 --cd /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof -- bash /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python -u /mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/bfm_entry250_labels_v1/source_snapshot_v1/audit_phase_compatibility.py --qualification $qualificationWsl 1>> (Join-Path $runRoot 'stdout.log') 2>> (Join-Path $runRoot 'stderr.log')
        $runExit=$LASTEXITCODE
    }
}catch{$runError=$_.Exception.Message}finally{
    $status.state='EXITED';$status['ended_utc']=[DateTime]::UtcNow.ToString('o');$status['process_exit_code']=$runExit;$status['launcher_error']=$runError
    $status['verdict']='Read labels and compatibility reports, then independent audit. No training authorized by this launcher.'
    $status | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $statePath -Encoding UTF8
}
exit $runExit
