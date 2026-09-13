param([double]$WallHours=3)
$ErrorActionPreference='Stop'
$repo='Z:\codex\GR00T-WholeBodyControl-sonic-transfer-23dof'
$python='C:\Users\camer\AppData\Local\Programs\Python\Python310\python.exe'
$stamp=Get-Date -Format 'yyyyMMdd_HHmmss'
$output=Join-Path 'E:\codex-artifacts\sonic23_teleop_resume_20260911\causal_dynamics_v1' "motion_curriculum_$stamp"
$env:PYTHONPATH=$repo
& $python (Join-Path $repo 'artifacts\teleop_resume_20260911\run_motion_curriculum.py') --output $output --wall-hours $WallHours
if($LASTEXITCODE -ne 0){throw "Run failed; see $output"}
Write-Output "Complete-motion results: $output"
