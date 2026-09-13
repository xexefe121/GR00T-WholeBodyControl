param([Parameter(Mandatory=$true)][string]$RequestSha256)
$ErrorActionPreference='Continue'
$runRoot='E:\codex-artifacts\sonic23_teleop_resume_20260911\filtered_all_candidates132_v1'
foreach($name in @('process_status.json','stdout.log','stderr.log','results')){if(Test-Path -LiteralPath (Join-Path $runRoot $name)){throw 'Preserve existing fixed132 diagnostic; no rerun.'}}
function Read-Sha256([string]$path){$hash=[Security.Cryptography.SHA256]::Create();try{return ([BitConverter]::ToString($hash.ComputeHash([IO.File]::ReadAllBytes($path)))).Replace('-','').ToLowerInvariant()}finally{$hash.Dispose()}}
$requestPath=Join-Path $runRoot 'request.json'
if((Read-Sha256 $requestPath) -ne $RequestSha256){throw 'Fixed request identity mismatch.'}
$request=Get-Content -LiteralPath $requestPath -Raw | ConvertFrom-Json
if($request.case_count -ne 132 -or $request.private_forecast_calls -ne 132 -or $request.connected_controller -ne $false -or $request.optimizer_calls -ne 0 -or $request.actor_calls -ne 0){throw 'Fixed diagnostic scope mismatch.'}
$status=[ordered]@{kind='one_fixed132_counterfactual_diagnostic';state='RUNNING';powershell_pid=$PID;started_utc=[DateTime]::UtcNow.ToString('o');request_sha256=$RequestSha256;cases=132;connected_controller=$false;optimizer_calls=0;actor_calls=0;hardware_authorized=$false}
$status | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $runRoot 'process_status.json') -Encoding UTF8
$runExit=-1;$runError=$null
try{
    & wsl.exe -d Ubuntu-22.04 --cd /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof -- bash /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python -u /mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/filtered_all_candidates132_v1/source_snapshot_v1/run_diagnostic.py 1> (Join-Path $runRoot 'stdout.log') 2> (Join-Path $runRoot 'stderr.log')
    $runExit=$LASTEXITCODE
}catch{$runError=$_.Exception.Message}finally{
    $status.state='EXITED';$status['ended_utc']=[DateTime]::UtcNow.ToString('o');$status['process_exit_code']=$runExit;$status['launcher_error']=$runError
    $status | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $runRoot 'process_status.json') -Encoding UTF8
}
exit $runExit
