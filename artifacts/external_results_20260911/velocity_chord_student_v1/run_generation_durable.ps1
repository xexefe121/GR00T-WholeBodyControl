param([Parameter(Mandatory=$true)][string]$ClearanceSha256,[Parameter(Mandatory=$true)][string]$FrozenReceiptSha256)
$ErrorActionPreference='Continue'
$runRoot='E:\codex-artifacts\sonic23_teleop_resume_20260911\velocity_chord_student_v1'
foreach($existing in @('generation_process_status.json','generation_stdout.log','generation_stderr.log','generation')) {
    if(Test-Path -LiteralPath (Join-Path $runRoot $existing)){throw 'Existing generation attempt must remain preserved.'}
}
function Read-Sha256([string]$path){
    $digest=[Security.Cryptography.SHA256]::Create()
    try{return ([BitConverter]::ToString($digest.ComputeHash([IO.File]::ReadAllBytes($path)))).Replace('-','').ToLowerInvariant()}finally{$digest.Dispose()}
}
$clearancePath=Join-Path $runRoot 'generation_clearance.json'
if((Read-Sha256 $clearancePath) -ne $ClearanceSha256){throw 'Generation clearance identity mismatch.'}
if((Read-Sha256 (Join-Path $runRoot 'generation_frozen_inputs.json')) -ne $FrozenReceiptSha256){throw 'Frozen generation identity mismatch.'}
$clearance=Get-Content -LiteralPath $clearancePath -Raw | ConvertFrom-Json
if($clearance.approved -ne $true -or $clearance.generation_only -ne $true){throw 'Clearance does not authorize bounded generation.'}
if($clearance.frozen_receipt_sha256 -ne $FrozenReceiptSha256){throw 'Clearance receipt binding mismatch.'}
if((Read-Sha256 (Join-Path $runRoot 'run_generation_durable.ps1')) -ne $clearance.launcher_sha256){throw 'Launcher identity mismatch.'}
if((Read-Sha256 $clearance.review_path) -ne $clearance.review_sha256){throw 'Review identity mismatch.'}
$status=[ordered]@{kind='one_fixed_velocity_chord_generation';state='RUNNING';powershell_pid=$PID;started_utc=[DateTime]::UtcNow.ToString('o');centers=3057;probes=140622;maximum_actor_calls=143679;maximum_backward_calls=3057;frozen_receipt_sha256=$FrozenReceiptSha256;clearance_sha256=$ClearanceSha256;optimizer_updates=0;physics_steps=0;hardware_authorized=$false}
$statePath=Join-Path $runRoot 'generation_process_status.json'
$status | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $statePath -Encoding UTF8
$rawExit=-1;$runExit=-1;$runError=$null
try{
    & wsl.exe -d Ubuntu-22.04 --cd /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof -- bash /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python -u /mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/velocity_chord_student_v1/source_snapshot_v1/generate_velocity_chords.py --clearance /mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/velocity_chord_student_v1/generation_clearance.json 1> (Join-Path $runRoot 'generation_stdout.log') 2> (Join-Path $runRoot 'generation_stderr.log')
    $rawExit=$LASTEXITCODE;$runExit=$rawExit
    if($rawExit -eq 0){
        $reportPath=Join-Path $runRoot 'generation/report.json'
        if(-not(Test-Path -LiteralPath $reportPath)){$runExit=5}else{
            $report=Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
            if($report.complete -ne $true -or $report.inference_calls.actor -ne 143679 -or $report.inference_calls.backward -ne 3057 -or $report.probe_rows -ne 140622 -or $report.all_center_byte_parity -ne $true){$runExit=5}
        }
    }
}catch{$runError=$_.Exception.Message;$runExit=7}finally{
    $status.state='EXITED';$status['ended_utc']=[DateTime]::UtcNow.ToString('o');$status['raw_python_exit_code']=$rawExit;$status['process_exit_code']=$runExit;$status['launcher_error']=$runError
    $status['verdict']='Generation evidence only. Fitting remains gated on independent dataset and source review.'
    $status | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $statePath -Encoding UTF8
}
exit $runExit
