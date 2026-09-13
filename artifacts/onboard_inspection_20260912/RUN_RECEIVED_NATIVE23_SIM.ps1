param(
    [ValidateSet('walk002', 'walk003', 'pico', 'walk008')]
    [string]$Clip = 'walk002',
    [string]$Actor,
    [switch]$Learned,
    [switch]$PicoDemo,
    [switch]$TaskCommands,
    [switch]$NativeTargets,
    [switch]$NativePreviewGuard,
    [ValidateSet('strict','diagnostic-only')]
    [string]$PreviewFailurePolicy = 'strict',
    [switch]$NativeStandingCapture,
    [ValidateRange(0,6)]
    [int]$NativePreviewDelaySubsteps = 0,
    [string]$NativePreviewLibrary,
    [string]$NativeClockLibrary,
    [switch]$FaultStandingCapture,
    [ValidateRange(0,500)]
    [int]$PlantSpinUs = 0,
    [switch]$IndependentClock,
    [switch]$Standing,
    [switch]$Lookahead,
    [ValidateRange(0.001,1.0)]
    [double]$TargetFilterAlpha = 1.0,
    [ValidateRange(10,100)]
    [int]$PredictionSteps = 30,
    [double]$InitialVx = 0.0,
    [int]$InputLossControl = -1,
    [switch]$Render,
    [switch]$FullQualityRender
)
$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$taskFirmware = 'E:\codex-artifacts\sonic23_teleop_resume_20260911\onboard_factory_firmware_v1'
$taskPython = 'C:\Users\camer\AppData\Local\Programs\Python\Python310\python.exe'
if ($NativePreviewGuard) { $IndependentClock = $true }
if ($PreviewFailurePolicy -eq 'diagnostic-only' -and -not $NativePreviewGuard) { throw 'Diagnostic-only requires -NativePreviewGuard.' }
if ($NativePreviewGuard -and -not $NativeTargets) { throw '-NativePreviewGuard requires -NativeTargets.' }
if ($NativeStandingCapture -and -not $NativeTargets) { throw '-NativeStandingCapture requires -NativeTargets.' }
if (($NativePreviewDelaySubsteps -ne 0 -or $NativePreviewLibrary) -and -not $NativePreviewGuard) { throw 'Native preview settings require -NativePreviewGuard.' }
if ($NativePreviewLibrary -and -not $IndependentClock) { throw '-NativePreviewLibrary requires -IndependentClock for the native Linux library.' }
if ($NativeClockLibrary -and -not $IndependentClock) { throw '-NativeClockLibrary requires -IndependentClock.' }
if ($TaskCommands -and (-not $Actor -or $Learned -or $PicoDemo)) { throw '-TaskCommands requires an explicit 1582-input -Actor.' }
if ($NativeTargets -and (-not $TaskCommands -or $Lookahead)) { throw '-NativeTargets requires -TaskCommands and a native-target actor; lookahead is unsupported.' }
if ($TaskCommands -and ($Lookahead -or $PSBoundParameters.ContainsKey('TargetFilterAlpha'))) { throw '-TaskCommands includes its own target filter; lookahead is unsupported.' }
if ($PicoDemo) {
    if ($Actor -or $Learned) { throw 'Choose -PicoDemo, -Learned, or an explicit -Actor.' }
    if ($PSBoundParameters.ContainsKey('TargetFilterAlpha')) { throw '-PicoDemo uses its recorded filter setting0.9.' }
    $Actor = Join-Path $taskFirmware 'received_pico_demo_v1\controller.onnx'
    $TargetFilterAlpha = 0.9
    $IndependentClock = $true
    $FaultStandingCapture = $true
    if (-not $PSBoundParameters.ContainsKey('Clip')) { $Clip = 'pico' }
}
if ($Learned) {
    if ($Actor) { throw 'Choose -Learned or an explicit -Actor.' }
    $Actor = Join-Path $taskFirmware 'locomotion_conditioned_ppo_v2\actor_00100.onnx'
    if (-not $PSBoundParameters.ContainsKey('Clip')) { $Clip = 'pico' }
}
$taskConditioned = [bool]$Actor
if (-not $Actor) { $Actor = Join-Path $taskFirmware 'human_loco_trainable_v1\factory_loco12.onnx' }
$Actor = (Resolve-Path -LiteralPath $Actor).Path
$taskStamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$taskOutput = Join-Path $taskFirmware "received_sim_runs\${taskStamp}_${Clip}"
$taskVelocity = $InitialVx.ToString([Globalization.CultureInfo]::InvariantCulture)
Write-Output 'Experimental full-body simulation. Full teleop acceptance remains unmet. No robot connection or motor publisher.'
if ($Standing -and -not $IndependentClock) { throw '-Standing requires -IndependentClock.' }
if ($FaultStandingCapture -and (-not $IndependentClock -or -not $taskConditioned)) { throw '-FaultStandingCapture requires an independent conditioned controller.' }
if ($PlantSpinUs -ne 0 -and -not $IndependentClock) { throw '-PlantSpinUs requires -IndependentClock.' }
if ($Lookahead -and -not $taskConditioned) { throw '-Lookahead requires an explicit 1542-input -Actor.' }
if ($TargetFilterAlpha -ne 1.0 -and -not $taskConditioned) { throw '-TargetFilterAlpha requires -Learned or an explicit conditioned -Actor.' }
if ($TargetFilterAlpha -ne 1.0 -and $Lookahead -and -not $IndependentClock) { throw 'Combined filtering and lookahead requires -IndependentClock.' }
if ($InputLossControl -ge 0 -and $InputLossControl -lt 1) { throw 'InputLossControl must follow control 0.' }

function Convert-TaskWslPath([string]$Path) {
    $taskFull = [IO.Path]::GetFullPath($Path)
    if ($taskFull -notmatch '^[A-Za-z]:\\') { throw 'Simulation paths must use a local drive letter.' }
    # E: is mounted in the launch command below. wslpath itself cannot resolve
    # that drive after WSL idles and drops the mount, so convert its known
    # /mnt/<drive> mapping without requiring an already-running distro.
    return '/mnt/' + $taskFull.Substring(0, 1).ToLowerInvariant() + '/' + $taskFull.Substring(3).Replace('\', '/')
}

if ($IndependentClock) {
    $taskClockLibrary = if ($NativeClockLibrary) { Convert-TaskWslPath (Resolve-Path -LiteralPath $NativeClockLibrary).Path } else { '/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/onboard_factory_firmware_v1/factory_clock_v3/libtrue23clock.so' }
    $taskMode = if ($TaskCommands) { '--task-commands' } elseif ($taskConditioned) { '--locomotion-conditioned' } else { '--factory-locomotion' }
    $taskScript = Convert-TaskWslPath (Join-Path $taskRoot 'artifacts\teleop_resume_20260911\run_causal_native_clock.py')
    $taskArgs = @('/mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python', $taskScript,
        '--actor', (Convert-TaskWslPath $Actor), '--output', (Convert-TaskWslPath $taskOutput),
        '--clip', $Clip, $taskMode, '--initial-velocity', $taskVelocity,
        '--library', $taskClockLibrary,
        '--affinity', '--realtime-priority')
    $taskArgs += @('--target-filter-alpha', $TargetFilterAlpha.ToString([Globalization.CultureInfo]::InvariantCulture))
    if ($NativeTargets) {
        $taskArgs += @('--native-targets', '--bank', '/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/causal_dynamics_v1/focused_walk002_task_closure_bank_v1')
    }
    if ($NativePreviewGuard) { $taskArgs += @('--native-preview-guard','--preview-failure-policy',$PreviewFailurePolicy) }
    if ($NativeStandingCapture) { $taskArgs += '--native-standing-capture' }
    if ($NativePreviewGuard) { $taskArgs += @('--native-preview-delay-substeps', $NativePreviewDelaySubsteps.ToString()) }
    if ($NativePreviewLibrary) { $taskArgs += @('--native-preview-library', (Convert-TaskWslPath (Resolve-Path -LiteralPath $NativePreviewLibrary).Path)) }
    if ($FaultStandingCapture) { $taskArgs += '--fault-standing-capture' }
    $taskArgs += @('--plant-spin-us', $PlantSpinUs.ToString())
    if ($Standing) { $taskArgs += @('--standing', '--controls', '1650') }
    if ($InputLossControl -ge 1) { $taskArgs += @('--fault-control', $InputLossControl.ToString()) }
    if ($Lookahead) { $taskArgs += @('--lookahead-library', '/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/onboard_factory_firmware_v1/native_shoot_v1/libtrue23shoot.so', '--lookahead-steps', $PredictionSteps.ToString()) }
    # User paths remain positional arguments, never interpolated into shell code.
    & wsl -d Ubuntu-22.04 -u root --exec bash -c 'mountpoint -q /mnt/e || mount -t drvfs E: /mnt/e; exec "$@"' -- @taskArgs
} elseif ($Lookahead) {
    if ($InputLossControl -ge 1) { throw 'Use -IndependentClock for input-loss lookahead tests.' }
    $taskArgs = @('/mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python',
        (Convert-TaskWslPath (Join-Path $PSScriptRoot 'run_native_shoot_sim.py')),
        '--actor', (Convert-TaskWslPath $Actor), '--output', (Convert-TaskWslPath $taskOutput),
        '--clip', $Clip, '--initial-vx', $taskVelocity, '--steps', $PredictionSteps.ToString(),
        '--library', '/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/onboard_factory_firmware_v1/native_shoot_v1/libtrue23shoot.so')
    & wsl -d Ubuntu-22.04 -u root --exec bash -c 'mountpoint -q /mnt/e || mount -t drvfs E: /mnt/e; exec "$@"' -- @taskArgs
} else {
    $taskMode = if ($TaskCommands) { '--task-commands' } elseif ($taskConditioned) { '--locomotion-conditioned' } else { '--factory-locomotion' }
    $taskArgs = @((Join-Path $PSScriptRoot 'run_factory_pose_sim.py'), '--clip', $Clip,
        '--actor', $Actor, '--output', $taskOutput, $taskMode, '--initial-vx', $taskVelocity)
    if ($NativeTargets) { $taskArgs += '--native-targets' }
    if ($NativePreviewGuard) { $taskArgs += '--native-preview-guard' }
    if ($NativeStandingCapture) { $taskArgs += '--native-standing-capture' }
    if ($NativePreviewGuard) { $taskArgs += @('--native-preview-delay-substeps', $NativePreviewDelaySubsteps.ToString()) }
    $taskArgs += @('--target-filter-alpha', $TargetFilterAlpha.ToString([Globalization.CultureInfo]::InvariantCulture))
    if ($InputLossControl -ge 1) { $taskArgs += @('--input-loss-control', $InputLossControl.ToString()) }
    & $taskPython @taskArgs
}
if ($LASTEXITCODE -ne 0) { throw "Simulation exited with code $LASTEXITCODE" }
Write-Output "Result: $taskOutput\report.json"
if ($Render) {
    $taskFps = if ($FullQualityRender) { '25' } else { '10' }
    $taskArgs = @((Join-Path $taskRoot 'artifacts\teleop_resume_20260911\render_native_clock.py'),
        '--run', $taskOutput, '--fps', $taskFps)
    if (-not $FullQualityRender) { $taskArgs += '--fast' }
    if ($IndependentClock) { $taskArgs += '--learned-clock' } else { $taskArgs += '--factory-trace' }
    & $taskPython @taskArgs
    if ($LASTEXITCODE -ne 0) { throw "Rendering exited with code $LASTEXITCODE" }
    $taskVideoFolder = if ($FullQualityRender) { 'video' } else { 'video_fast' }
    Write-Output "Video: $taskOutput\$taskVideoFolder"
}
