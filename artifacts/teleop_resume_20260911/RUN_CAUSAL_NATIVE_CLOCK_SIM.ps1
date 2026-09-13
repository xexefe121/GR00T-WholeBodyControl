param(
    [Parameter(Mandatory=$true)][string]$Actor,
    [ValidateSet('walk002','walk003','pico','walk008')][string]$Clip='walk002',
    [switch]$Standing,
    [int]$Controls=0,
    [double]$InitialVelocity=0,
    [int]$FaultControl=-1
)
$ErrorActionPreference='Stop'
$resolvedActor=(Resolve-Path -LiteralPath $Actor).Path.Replace('\','/')
if($resolvedActor -notmatch '^([A-Za-z]):/(.*)$'){throw 'Actor must be a local Windows drive path.'}
$linuxActor='/mnt/'+$Matches[1].ToLower()+'/'+$Matches[2]
$stamp=Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$folder="manual_$stamp"
$localOutput="E:\codex-artifacts\sonic23_teleop_resume_20260911\causal_native_clock_v1\$folder"
$linuxOutput="/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/causal_native_clock_v1/$folder"
$repo='/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof'
$runner="$repo/artifacts/teleop_resume_20260911/run_causal_native_clock.py"
$bootstrap="$repo/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh"
$python='/mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python'
$arguments=@('--actor',$linuxActor,'--clip',$Clip,'--initial-velocity',[string]$InitialVelocity,'--output',$linuxOutput,'--affinity')
if($Standing){$arguments+='--standing'}
if($Controls -gt 0){$arguments+=@('--controls',[string]$Controls)}
if($FaultControl -ge 0){$arguments+=@('--fault-control',[string]$FaultControl)}
& wsl.exe -d Ubuntu-22.04 --cd / -- bash $bootstrap env "PYTHONPATH=$repo" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= $python $runner @arguments
if($LASTEXITCODE -ne 0){throw "Simulation process failed. Output: $localOutput"}
$report=Get-Content -LiteralPath (Join-Path $localOutput 'report.json') -Raw | ConvertFrom-Json
Write-Output "Simulation report: $localOutput\report.json"
$scenarioPassed=if($FaultControl -ge 0){$report.fault_scenario_passed}else{$report.passed}
if(-not $scenarioPassed){throw 'Controller failed behavior, packet, physical or timing criteria. See report.'}
Write-Output 'Requested simulation scenario passed. General teleoperation remains unqualified.'
