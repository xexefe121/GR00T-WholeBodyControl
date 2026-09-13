param([Parameter(Mandatory=$true)][string]$ClearanceSha256)
$ErrorActionPreference='Continue'
$runRoot='E:\codex-artifacts\sonic23_teleop_resume_20260911\fast_controller_cost_ranked_v1'
foreach($existing in @('canonical_process_status.json','canonical_stdout.log','canonical_stderr.log','nominal','post_lifecycle_hold_5s')){
    if(Test-Path -LiteralPath (Join-Path $runRoot $existing)){throw 'Existing cost-ranked canonical attempt must remain preserved.'}
}
function Read-Sha256([string]$path){
    $digest=[Security.Cryptography.SHA256]::Create()
    try{return ([BitConverter]::ToString($digest.ComputeHash([IO.File]::ReadAllBytes($path)))).Replace('-','').ToLowerInvariant()}finally{$digest.Dispose()}
}
$clearancePath=Join-Path $runRoot 'runtime_clearance.json'
if((Read-Sha256 $clearancePath) -ne $ClearanceSha256){throw 'Cost-ranked runtime clearance identity mismatch.'}
$clearance=Get-Content -LiteralPath $clearancePath -Raw | ConvertFrom-Json
if($clearance.canonical_rollout_authorized -ne $true -or $clearance.canonical_trials -ne 1 -or $clearance.private_horizon_controls -ne 5 -or $clearance.actual_commit_controls -ne 1){throw 'Cost-ranked runtime clearance scope mismatch.'}
if($clearance.root_selection_pending -ne $false -or $clearance.source_review_pending -ne $false){throw 'Root selection and final source review remain required.'}
if($clearance.selection -ne 'lowest_exact_feasible_prefix_cost' -or $clearance.evaluate_all_four -ne $true){throw 'All-four exact feasible cost ranking is required.'}
if($clearance.head_sha256 -ne '861b4c39349276851e23edab43978a74bab4cc613915da87805471c6164bc365'){throw 'Unchanged final65000 head is required.'}
foreach($required in @('frozen_inputs_v2.json','run_canonical_durable.ps1','root_selection.json','source_snapshot_v1\evaluate_filtered_student.py','source_snapshot_v1\native_forecast.py','source_snapshot_v1\transactional_student.py','source_snapshot_v1\ranked_admission.py','source_snapshot_v1\prefix_tracking_cost.py','saved132_cost_equivalence_test_report.json')){
    $requiredPath=[IO.Path]::GetFullPath((Join-Path $runRoot $required))
    $bindings=@($clearance.bound_files | Where-Object {[IO.Path]::GetFullPath($_.path) -ieq $requiredPath})
    if($bindings.Count -ne 1){throw ('Required runtime input not bound exactly once: '+$required)}
}
foreach($item in $clearance.bound_files){if((Read-Sha256 $item.path) -ne $item.sha256){throw ('Reviewed cost-ranked runtime input changed: '+$item.path)}}
if((Read-Sha256 (Join-Path $runRoot 'frozen_inputs_v2.json')) -ne $clearance.frozen_sources_sha256){throw 'Frozen cost-ranked sources changed.'}
$statePath=Join-Path $runRoot 'canonical_process_status.json'
$status=[ordered]@{kind='one_synchronous_cost_ranked65000_student';state='RUNNING';powershell_pid=$PID;started_utc=[DateTime]::UtcNow.ToString('o');requested_controls=1569;separate_hold_controls=250;private_horizon_controls=5;actual_commit_controls=1;initial_BFM_controls=250;head_sha256=$clearance.head_sha256;clearance_sha256=$ClearanceSha256;additional_fitting=$false;expert_query=$false;hardware_authorized=$false}
$status | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $statePath -Encoding UTF8
$runExit=-1;$pythonExit=-1;$runError=$null;$lifecycleComplete=$false;$lifecycleError=$null
try{
    & wsl.exe -d Ubuntu-22.04 --cd /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof -- bash /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python -u /mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/fast_controller_cost_ranked_v1/source_snapshot_v1/evaluate_filtered_student.py 1> (Join-Path $runRoot 'canonical_stdout.log') 2> (Join-Path $runRoot 'canonical_stderr.log')
    $pythonExit=$LASTEXITCODE;$runExit=$pythonExit
    if($pythonExit -eq 0){
        try{
            $outcome=Get-Content -LiteralPath (Join-Path $runRoot 'pilot_outcome.json') -Raw -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop
            $lifecycleComplete=($outcome.nominal.full_segment_completed -eq $true -and $outcome.extension.full_segment_completed -eq $true -and $null -eq $outcome.nominal.failure -and $null -eq $outcome.extension.failure -and $outcome.nominal.quiet_standing_diagnostic.quiet_standing_diagnostic_pass -eq $true -and $outcome.extension.quiet_standing_diagnostic.quiet_standing_diagnostic_pass -eq $true)
            if(-not $lifecycleComplete){$runExit=5;$lifecycleError='Saved lifecycle or continuous hold incomplete, physically failed, or failed quiet standing.'}
        }catch{$runExit=6;$lifecycleError='Missing or unreadable completed lifecycle outcome: '+$_.Exception.Message}
    }
}catch{$runError=$_.Exception.Message}finally{
    $status.state='EXITED';$status['ended_utc']=[DateTime]::UtcNow.ToString('o');$status['process_exit_code']=$runExit;$status['launcher_error']=$runError
    $status['raw_python_exit_code']=$pythonExit;$status['lifecycle_and_hold_completed_with_quiet']=$lifecycleComplete;$status['lifecycle_error']=$lifecycleError
    $status['verdict']='Nonzero exit preserves incomplete/failed lifecycle. Even zero exit requires independent actual/forecast/transition/source/quiet/timing qualification.'
    $status | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $statePath -Encoding UTF8
}
exit $runExit
