param(
    [Parameter(Mandatory=$true)][string]$Actor,
    [ValidateSet('walk003','walk002','pico','walk008')][string[]]$Clips=@('walk003','walk002','pico','walk008'),
    [double]$InitialVelocity=0,
    [int]$FaultControl=-1,
    [switch]$Balanced
)
$ErrorActionPreference='Stop'
$python='C:\Users\camer\AppData\Local\Programs\Python\Python310\python.exe'
$repo='Z:\codex\GR00T-WholeBodyControl-sonic-transfer-23dof'
$resolved=(Resolve-Path -LiteralPath $Actor).Path
$stamp=Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$output=Join-Path 'E:\codex-artifacts\sonic23_teleop_resume_20260911\causal_dynamics_v1' "manual_$stamp"
$env:PYTHONPATH=$repo
$arguments=@('-m','gear_sonic.scripts.evaluate_g1_true23_causal_dynamics','--actor',$resolved,'--output',$output,
             '--initial-velocity',([string]$InitialVelocity),'--hold','30','--clips')+$Clips
if($FaultControl -ge 0){$arguments+=@('--fault-control',([string]$FaultControl))}
$requestPath=Join-Path (Split-Path -Parent $resolved) 'request.json'
$trainedWithBalance=$false
if(Test-Path -LiteralPath $requestPath){$trainedWithBalance=(Get-Content -LiteralPath $requestPath -Raw | ConvertFrom-Json).retained_neutral_balance -eq $true}
if($Balanced -or $trainedWithBalance){$arguments+='--balanced'}
& $python @arguments
if($LASTEXITCODE -ne 0){throw "Simulation process failed. Output: $output"}
$report=Get-Content -LiteralPath (Join-Path $output 'report.json') -Raw | ConvertFrom-Json
Write-Output "Simulation report: $output\report.json"
if(-not $report.passed){throw 'Controller did not pass all full-motion and standing checks. See report; no readiness claim.'}
