param([double]$InitialVelocity=0)
$ErrorActionPreference='Stop'
$stamp=Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$folder="manual_$stamp"
$localOutput="E:\codex-artifacts\sonic23_teleop_resume_20260911\native_clock_v1\$folder"
$linuxOutput="/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/native_clock_v1/$folder"
$repo='/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof'
$runner="$repo/artifacts/teleop_resume_20260911/run_native_clock_isolated.py"
$bootstrap="$repo/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh"
$python='/mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python'
& wsl.exe -d Ubuntu-22.04 --cd / -- bash $bootstrap env "PYTHONPATH=$repo" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $python $runner --braking --initial-velocity ([string]$InitialVelocity) --output $linuxOutput
if($LASTEXITCODE -ne 0){throw "Simulation process failed. Output: $localOutput"}
$report=Get-Content -LiteralPath (Join-Path $localOutput 'report.json') -Raw | ConvertFrom-Json
Write-Output "Simulation report: $localOutput\report.json"
if(-not $report.prepared_benchmark_pass){throw 'Prepared controller failed motion, standing or timing criteria. See report.'}
Write-Output 'Prepared walk003 benchmark passed. General live teleoperation remains unqualified.'
