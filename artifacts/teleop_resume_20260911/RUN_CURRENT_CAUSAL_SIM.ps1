param(
    [ValidateSet('walk002','walk003','pico','walk008')][string]$Clip='walk002',
    [switch]$Render,
    [switch]$Independent,
    [string]$Campaign='E:\codex-artifacts\sonic23_teleop_resume_20260911\causal_dynamics_v1\motion_curriculum_run_v2'
)
$ErrorActionPreference='Stop'
$repo='Z:\codex\GR00T-WholeBodyControl-sonic-transfer-23dof'
$python='C:\Users\camer\AppData\Local\Programs\Python\Python310\python.exe'
$base='E:\codex-artifacts\sonic23_teleop_resume_20260911\causal_dynamics_v1'
$selection=Join-Path $Campaign 'best_motion.json'
if(Test-Path -LiteralPath $selection){
    $actor=(Get-Content -Raw -LiteralPath $selection | ConvertFrom-Json).actor
}else{
    $actor=Join-Path $base 'task_closure_v2_candidate\actor.onnx'
}
$actor=(Resolve-Path -LiteralPath $actor).Path
$output=Join-Path $base ('current_sim_'+(Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
$bank=Join-Path $base 'focused_walk002_task_closure_bank_v1'
$env:PYTHONPATH=$repo
Write-Output "Simulating $Clip with $actor"
Write-Output 'Development controller. Complete-motion accuracy and live timing are still under evaluation.'
if($Independent){
    function Convert-ToWslPath([string]$Path){
        $normalized=[System.IO.Path]::GetFullPath($Path).Replace('\','/')
        if($normalized -notmatch '^([A-Za-z]):/(.*)$'){throw 'Expected a local Windows drive path.'}
        return '/mnt/'+$Matches[1].ToLower()+'/'+$Matches[2]
    }
    $linuxRepo=Convert-ToWslPath $repo
    $linuxActor=Convert-ToWslPath $actor
    $linuxBank=Convert-ToWslPath $bank
    $linuxOutput=Convert-ToWslPath $output
    $bootstrap="$linuxRepo/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh"
    $runner="$linuxRepo/artifacts/teleop_resume_20260911/run_causal_native_clock.py"
    $linuxPython='/mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python'
    & wsl.exe -d Ubuntu-22.04 --cd / -- bash $bootstrap env "PYTHONPATH=$linuxRepo" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= $linuxPython $runner --actor $linuxActor --bank $linuxBank --clip $Clip --hold 30 --output $linuxOutput --affinity
}else{
    & $python -m gear_sonic.scripts.evaluate_g1_true23_causal_dynamics --actor $actor --bank $bank --output $output --clips $Clip --hold 30
}
if($LASTEXITCODE -ne 0){throw "Simulation process failed; see $output"}
$report=Get-Content -Raw -LiteralPath (Join-Path $output 'report.json') | ConvertFrom-Json
if($Render){
    $renderRun=if($Independent){$output}else{Join-Path $output $Clip}
    $renderKind=if($Independent){'--learned-clock'}else{'--causal-trace'}
    & $python (Join-Path $repo 'artifacts\teleop_resume_20260911\render_native_clock.py') --run $renderRun $renderKind
    if($LASTEXITCODE -ne 0){throw "Video rendering failed; simulation results remain in $output"}
    $videoName=if($Independent){"${Clip}_learned_independent_clock.mp4"}else{"${Clip}_learned_evaluation.mp4"}
    Write-Output ('Video: '+(Join-Path $renderRun "video\$videoName"))
}
Write-Output "Simulation report: $output\report.json"
Write-Output "Requested full-motion and standing criteria passed: $($report.passed)"
if($Independent){
    Write-Output "Independent500Hz physics /50Hz control; missed control deadlines: $($report.controller_deadline_misses)"
}else{
    Write-Output 'This run uses unpaced physics.'
}
Write-Output 'General live teleoperation is not qualified by this result.'
if(-not $report.passed){exit 2}
