param(
    [ValidateSet('walk002','walk003','pico','walk008')][string]$Clip='walk002',
    [ValidateSet('recorded','live')][string]$InputMode='recorded',
    [ValidateSet('ground-truth','estimated')][string]$Feedback='ground-truth',
    [string]$Actor,
    [string]$SensorEffects,
    [string]$LiveUrl='tcp://127.0.0.1:5557',
    [string]$PacketReplay,
    [string]$RearmFile,
    [ValidateRange(0,100000)][int]$Controls=0,
    [ValidateRange(-0.03,0.03)][double]$InitialVx=0.,
    [int]$InputLossControl=-1,
    [switch]$Render
)
$ErrorActionPreference='Stop'
$taskRoot=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$taskFirmware='E:\codex-artifacts\sonic23_teleop_resume_20260911\onboard_factory_firmware_v1'
if (-not $Actor) {$Actor=Join-Path $taskFirmware 'controller_state_ab_v1\retained_controller\controller.onnx'}
$Actor=(Resolve-Path -LiteralPath $Actor).Path
$taskStamp=Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$taskOutput=Join-Path $taskFirmware "received_sim_runs\${taskStamp}_${Clip}_${InputMode}_${Feedback}"
function Convert-TaskPath([string]$Path) {
    $taskFull=[IO.Path]::GetFullPath($Path)
    if ($taskFull -notmatch '^[A-Za-z]:\\') {throw 'Expected a local drive path.'}
    return '/mnt/'+$taskFull.Substring(0,1).ToLowerInvariant()+'/'+$taskFull.Substring(3).Replace('\','/')
}
$taskArgs=@('/mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python',
    (Convert-TaskPath (Join-Path $taskRoot 'artifacts\teleop_resume_20260911\run_causal_native_clock.py')),
    '--actor',(Convert-TaskPath $Actor),'--output',(Convert-TaskPath $taskOutput),'--clip',$Clip,
    '--task-commands','--native-targets','--native-preview-guard','--native-standing-capture',
    '--native-preview-delay-substeps','6','--input',$InputMode,'--feedback',$Feedback,
    '--bank','/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/causal_dynamics_v1/focused_walk002_task_closure_bank_v1',
    '--library','/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/onboard_factory_firmware_v1/factory_clock_v5/libtrue23clock.so',
    '--native-preview-library','/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/onboard_factory_firmware_v1/native_preview_backend_v3/libtrue23preview.so',
    '--initial-velocity',$InitialVx.ToString([Globalization.CultureInfo]::InvariantCulture),
    '--affinity','--realtime-priority')
if ($Controls) {$taskArgs+=@('--controls',$Controls.ToString())}
if ($PacketReplay) {$taskArgs+=@('--packet-replay',(Convert-TaskPath (Resolve-Path -LiteralPath $PacketReplay).Path))}
if ($InputLossControl -ge 1) {$taskArgs+=@('--fault-control',$InputLossControl.ToString())}
if ($SensorEffects) {$taskArgs+=@('--sensor-effects',(Convert-TaskPath (Resolve-Path -LiteralPath $SensorEffects).Path))}
if ($InputMode -eq 'live') {
    $taskArgs+=@('--live-url',$LiveUrl)
    if ($RearmFile) {$taskArgs+=@('--rearm-file',(Convert-TaskPath $RearmFile))}
}
Write-Output "Simulation candidate: $Actor"
Write-Output "Input: $InputMode; feedback: $Feedback. Full teleop qualification remains incomplete."
& wsl -d Ubuntu-22.04 -u root --exec bash -c 'mountpoint -q /mnt/e || mount -t drvfs E: /mnt/e; exec "$@"' -- @taskArgs
if ($LASTEXITCODE -ne 0) {throw "Simulation exited with code $LASTEXITCODE. Inspect $taskOutput"}
Write-Output "Result: $taskOutput\report.json"
if ($Render) {
    if ($InputMode -ne 'recorded') {throw 'Live packets need their own reference evaluator before comparison rendering.'}
    & 'C:\Users\camer\AppData\Local\Programs\Python\Python310\python.exe' (Join-Path $taskRoot 'artifacts\teleop_resume_20260911\render_native_clock.py') --run $taskOutput --fps 25 --learned-clock
    if ($LASTEXITCODE -ne 0) {throw 'Video rendering failed.'}
}
