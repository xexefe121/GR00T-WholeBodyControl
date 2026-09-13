param([switch]$Paced)
$ErrorActionPreference='Stop'
$taskBase='E:/codex-artifacts/sonic23_teleop_resume_20260911/fast_feedback_walk003_v1'
$runName='user_trial_'+[DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
$scriptName=if($Paced){'run_paced_controller.py'}else{'run_controller.py'}
$linuxBase='/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/fast_feedback_walk003_v1'
$arguments=@('-d','Ubuntu-22.04','--cd','/','--','bash',
    '/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh',
    'env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','PYTHONDONTWRITEBYTECODE=1',
    'PYTHONPATH=/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof',
    '/mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python','-B','-u',
    ($linuxBase+'/'+$scriptName),'--output',($linuxBase+'/'+$runName))
Write-Host 'Fresh native23 simulation: prepared walk003 state-feedback controller. No robot commands.'
& "$env:SystemRoot\System32\wsl.exe" @arguments
if($LASTEXITCODE -ne 0){throw "Simulation process failed. Preserve $taskBase/$runName for diagnosis."}
$reportPath=$taskBase+'/'+$runName+'/report.json'
$report=Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
Write-Host ('Report: '+$reportPath)
if(-not $report.full_motion_and_hold_completed){throw 'Full motion and standing hold did not complete. See report.'}
if(-not $report.main.quiet_standing_diagnostic.quiet_standing_diagnostic_pass -or -not $report.hold.quiet_standing_diagnostic.quiet_standing_diagnostic_pass){throw 'Motion completed, but the original quiet-standing acceptance failed. See report.'}
$metrics=$report.main.source_metrics
if($metrics.source_controls -ne 819 -or $metrics.original_root_world_p95_m -gt .20 -or $metrics.original_root_yaw_abs_p95_deg -gt 15 -or $metrics.leg_rmse_rad -gt .15 -or $metrics.original_hand_head_relative_p95_m[0] -gt .15 -or $metrics.original_hand_head_relative_p95_m[1] -gt .15 -or $metrics.original_hand_head_relative_p95_m[2] -gt .10 -or $metrics.world_axis_relative_foot_p95_m[0] -gt .12 -or $metrics.world_axis_relative_foot_p95_m[1] -gt .12){throw 'Original full-source tracking acceptance failed. See report.'}
Write-Host 'Full motion and continuous hold completed. Motion-specific simulation baseline; live teleoperation remains unqualified.'
