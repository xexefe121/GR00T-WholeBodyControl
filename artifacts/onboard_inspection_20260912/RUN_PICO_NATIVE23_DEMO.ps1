param(
    [switch]$Render,
    [switch]$FullQualityRender,
    [double]$InitialVx = 0.0,
    [int]$InputLossControl = -1
)
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'RUN_RECEIVED_NATIVE23_SIM.ps1') -PicoDemo -Render:$Render -FullQualityRender:$FullQualityRender -InitialVx $InitialVx -InputLossControl $InputLossControl
