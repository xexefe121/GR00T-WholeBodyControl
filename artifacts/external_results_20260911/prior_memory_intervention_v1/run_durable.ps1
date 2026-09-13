param([Parameter(Mandatory=$true)][string]$ClearanceSha256)
$ErrorActionPreference='Continue'
$runRoot='E:\codex-artifacts\sonic23_teleop_resume_20260911\prior_memory_intervention_v1'
foreach($name in @('process_status.json','stdout.log','stderr.log','results','preflight_failure.json')){
    if(Test-Path -LiteralPath (Join-Path $runRoot $name)){throw 'Existing diagnostic attempt must be preserved.'}
}
function Read-TaskHash([string]$path){
    $digest=[Security.Cryptography.SHA256]::Create()
    try{return ([BitConverter]::ToString($digest.ComputeHash([IO.File]::ReadAllBytes($path)))).Replace('-','').ToLowerInvariant()}finally{$digest.Dispose()}
}
$clearancePath=Join-Path $runRoot 'clearance.json'
if((Read-TaskHash $clearancePath) -ne $ClearanceSha256){throw 'Clearance hash mismatch.'}
$clearance=Get-Content -LiteralPath $clearancePath -Raw | ConvertFrom-Json
if($clearance.approved -ne $true -or $clearance.total_graph_calls -ne 60){throw 'Diagnostic scope mismatch.'}
if((Read-TaskHash (Join-Path $runRoot 'request.json')) -ne $clearance.request_sha256){throw 'Request mismatch.'}
if((Read-TaskHash (Join-Path $runRoot 'run_durable.ps1')) -ne $clearance.launcher_sha256){throw 'Launcher mismatch.'}
if((Read-TaskHash $clearance.review_path) -ne $clearance.review_sha256){throw 'Review mismatch.'}
$status=[ordered]@{state='RUNNING';powershell_pid=$PID;started_utc=[DateTime]::UtcNow.ToString('o');clearance_sha256=$ClearanceSha256;expected_backward_calls=12;expected_actor_calls=24;expected_head_calls=24;physics_steps=0;optimizer_updates=0}
$statusPath=Join-Path $runRoot 'process_status.json'
$status | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8
$rawExit=-1;$exitCode=-1;$launchError=$null
try{
    & wsl.exe -d Ubuntu-22.04 --cd /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof -- bash /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python /mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/prior_memory_intervention_v1/source_snapshot_v2/run_diagnostic.py --clearance-sha256 $ClearanceSha256 1> (Join-Path $runRoot 'stdout.log') 2> (Join-Path $runRoot 'stderr.log')
    $rawExit=$LASTEXITCODE
    if($null -eq $rawExit){$rawExit=-2;throw 'WSL exit code unavailable.'}
    $exitCode=$rawExit
    if($rawExit -eq 0){
        $reportPath=Join-Path $runRoot 'results/report.json'
        if(-not(Test-Path -LiteralPath $reportPath)){$exitCode=5}else{
            $report=Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
            if($report.passed -ne $true -or $report.total_graph_calls -ne 60){$exitCode=5}
        }
    }
}catch{$launchError=$_.Exception.Message;$exitCode=7}finally{
    $status.state='EXITED';$status['ended_utc']=[DateTime]::UtcNow.ToString('o');$status['raw_wsl_exit_code']=$rawExit;$status['process_exit_code']=$exitCode;$status['launcher_error']=$launchError
    $status | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8
}
exit $exitCode
