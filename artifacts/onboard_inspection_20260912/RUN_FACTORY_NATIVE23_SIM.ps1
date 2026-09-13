param(
    [switch]$Render,
    [double]$InitialVx = 0.0
)
$ErrorActionPreference = 'Stop'
$taskPython = 'C:\Users\camer\AppData\Local\Programs\Python\Python310\python.exe'
$taskScript = Join-Path $PSScriptRoot 'run_factory_mimic_sim.py'
$taskOutput = 'E:\codex-artifacts\sonic23_teleop_resume_20260911\onboard_factory_firmware_v1\native23_factory_demo_latest'
$taskArgs = @($taskScript, '--joint-margin', '0.06', '--limit-brake', '--initial-vx', $InitialVx.ToString([Globalization.CultureInfo]::InvariantCulture), '--output', $taskOutput)
if ($Render) { $taskArgs += '--render' }
& $taskPython @taskArgs
if ($LASTEXITCODE -ne 0) { throw "Simulation exited with code $LASTEXITCODE" }
Write-Output "Simulation result: $taskOutput\report.json"
if ($Render) { Write-Output "Full-duration video: $taskOutput\factory_native23.mp4" }
