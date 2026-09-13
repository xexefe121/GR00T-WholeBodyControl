param([Parameter(Mandatory=$true)][string]$ClearanceSha256)
$ErrorActionPreference='Continue'
$runRoot='E:\codex-artifacts\sonic23_teleop_resume_20260911\fast_controller_phase_fit_v1'
foreach($existing in @('canonical_process_status.json','canonical_stdout.log','canonical_stderr.log','nominal','post_lifecycle_hold_5s')){
    if(Test-Path -LiteralPath (Join-Path $runRoot $existing)){throw 'Existing phase canonical attempt must remain preserved.'}
}
function Read-Sha256([string]$path){
    $digest=[Security.Cryptography.SHA256]::Create()
    try{return ([BitConverter]::ToString($digest.ComputeHash([IO.File]::ReadAllBytes($path)))).Replace('-','').ToLowerInvariant()}finally{$digest.Dispose()}
}
$clearancePath=Join-Path $runRoot 'final_export_clearance.json'
if((Read-Sha256 $clearancePath) -ne $ClearanceSha256){throw 'Final phase export clearance identity mismatch.'}
$clearance=Get-Content -LiteralPath $clearancePath -Raw | ConvertFrom-Json
if($clearance.canonical_rollout_authorized -ne $true -or $clearance.ordinary_final_global_step -ne 65000 -or $clearance.additional_updates -ne 5000){throw 'Final clearance scope mismatch.'}
if($clearance.export_parity_passed -ne $true -or $clearance.full_objective_improved -ne $true){throw 'Ordinary-final numerical gates did not pass.'}
foreach($required in @('fit\student_head.onnx','fit\student_head.pt','fit\report.json','fit\restoration60000_parity.json','fit\optimization_completed.json','final_export_validation.json','frozen_inputs_v2.json','run_canonical_durable.ps1')){
    $requiredPath=[IO.Path]::GetFullPath((Join-Path $runRoot $required))
    $bindings=@($clearance.bound_files | Where-Object {[IO.Path]::GetFullPath($_.path) -ieq $requiredPath})
    if($bindings.Count -ne 1){throw ('Required final input not bound exactly once: '+$required)}
}
foreach($item in $clearance.bound_files){if((Read-Sha256 $item.path) -ne $item.sha256){throw ('Reviewed final input changed: '+$item.path)}}
if((Read-Sha256 (Join-Path $runRoot 'frozen_inputs_v2.json')) -ne $clearance.frozen_sources_sha256){throw 'Frozen phase source changed.'}
$fit=Get-Content -LiteralPath (Join-Path $runRoot 'fit\report.json') -Raw | ConvertFrom-Json
if($fit.network_training_complete -ne $true -or $fit.steps -ne 65000 -or $fit.additional_updates -ne 5000 -or $fit.export_parity_passed -ne $true -or $fit.full_objective_improved -ne $true -or $fit.rollout_numerical_prerequisites_pass -ne $true -or $fit.ONNX_vs_Torch_max_delta_rad -ge 0.00001){throw 'Final phase fit/export report did not pass.'}
$statePath=Join-Path $runRoot 'canonical_process_status.json'
$status=[ordered]@{kind='one_canonical_phase_student_rollout';state='RUNNING';powershell_pid=$PID;started_utc=[DateTime]::UtcNow.ToString('o');requested_controls=1569;separate_hold_controls=250;fresh_canonical_BFM_prefix_controls=250;learned_controls=1019;cached_expert_commands_used=$false;hardware_authorized=$false;clearance_sha256=$ClearanceSha256}
$status | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $statePath -Encoding UTF8
$runExit=-1;$runError=$null
try{
    & wsl.exe -d Ubuntu-22.04 --cd /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof -- bash /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python -u /mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/fast_controller_phase_fit_v1/source_snapshot_v1/evaluate_phase_student.py 1> (Join-Path $runRoot 'canonical_stdout.log') 2> (Join-Path $runRoot 'canonical_stderr.log')
    $runExit=$LASTEXITCODE
}catch{$runError=$_.Exception.Message}finally{
    $status.state='EXITED';$status['ended_utc']=[DateTime]::UtcNow.ToString('o');$status['process_exit_code']=$runExit;$status['launcher_error']=$runError
    $status['physical_verdict']='Read transition parity, nominal, hold and independent audits; process exit is not qualification.'
    $status | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $statePath -Encoding UTF8
}
exit $runExit
