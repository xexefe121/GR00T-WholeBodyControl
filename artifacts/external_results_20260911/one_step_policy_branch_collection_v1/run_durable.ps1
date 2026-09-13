param([Parameter(Mandatory=$true)][string]$ClearanceSha256)
$ErrorActionPreference='Continue'
$runRoot='E:\codex-artifacts\sonic23_teleop_resume_20260911\one_step_policy_branch_collection_v1'
foreach($name in @('process_status.json','stdout.log','stderr.log','collection','preflight_failure.json')){
    if(Test-Path -LiteralPath (Join-Path $runRoot $name)){throw 'Existing collection attempt must be preserved.'}
}
function Read-TaskHash([string]$path){
    $digest=[Security.Cryptography.SHA256]::Create()
    try{return ([BitConverter]::ToString($digest.ComputeHash([IO.File]::ReadAllBytes($path)))).Replace('-','').ToLowerInvariant()}finally{$digest.Dispose()}
}
$clearancePath=Join-Path $runRoot 'clearance.json'
if((Read-TaskHash $clearancePath) -ne $ClearanceSha256){throw 'Clearance hash mismatch.'}
$clearance=Get-Content -LiteralPath $clearancePath -Raw | ConvertFrom-Json
if($clearance.approved -ne $true -or $clearance.rows -ne 3054 -or $clearance.native_step_ceiling -ne 61080 -or $clearance.total_graph_call_ceiling -ne 12216){throw 'Selected collection scope mismatch.'}
if((Read-TaskHash (Join-Path $runRoot 'request.json')) -ne $clearance.request_sha256){throw 'Request mismatch.'}
if((Read-TaskHash (Join-Path $runRoot 'run_durable.ps1')) -ne $clearance.launcher_sha256){throw 'Launcher mismatch.'}
if((Read-TaskHash $clearance.review_path) -ne $clearance.review_sha256){throw 'Review mismatch.'}
$status=[ordered]@{state='RUNNING';powershell_pid=$PID;started_utc=[DateTime]::UtcNow.ToString('o');clearance_sha256=$ClearanceSha256;rows=3054;native_step_ceiling=61080;total_graph_call_ceiling=12216;optimizer_updates=0}
$statusPath=Join-Path $runRoot 'process_status.json'
$status | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8
$rawExit=-1;$exitCode=-1;$launchError=$null
try{
    & wsl.exe -d Ubuntu-22.04 --cd /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof -- bash /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python /mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/one_step_policy_branch_collection_v1/source_snapshot_v1/collect_branches.py --clearance-sha256 $ClearanceSha256 1> (Join-Path $runRoot 'stdout.log') 2> (Join-Path $runRoot 'stderr.log')
    $rawExit=$LASTEXITCODE
    if($null -eq $rawExit){$rawExit=-2;throw 'WSL exit code unavailable.'}
    $exitCode=$rawExit
    if($rawExit -eq 0){
        $reportPath=Join-Path $runRoot 'collection/report.json'
        if(-not(Test-Path -LiteralPath $reportPath)){$exitCode=5}else{
            $report=Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
            if($report.completed -ne $true -or $report.passed -ne $true -or $report.rows -ne 3054 -or $report.nominal_verified -ne 3054 -or $report.policy_branches -ne 3054 -or $report.query250_actual251_calibration_passed -ne $true){$exitCode=5}
        }
    }
}catch{$launchError=$_.Exception.Message;$exitCode=7}finally{
    $status.state='EXITED';$status['ended_utc']=[DateTime]::UtcNow.ToString('o');$status['raw_wsl_exit_code']=$rawExit;$status['process_exit_code']=$exitCode;$status['launcher_error']=$launchError
    $status | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8
}
exit $exitCode
