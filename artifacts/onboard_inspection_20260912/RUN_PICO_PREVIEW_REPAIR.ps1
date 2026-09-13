param(
    [ValidateSet('older','strict-newer')]
    [string]$Case = 'older'
)
$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$taskScript = (Join-Path $taskRoot 'artifacts\onboard_inspection_20260912\run_pico_preview_repair.py').Replace('\','/')
$taskDrive = $taskScript.Substring(0,1).ToLowerInvariant()
$taskScript = '/mnt/' + $taskDrive + $taskScript.Substring(2)
$taskStamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$taskOutput = "/mnt/e/codex-artifacts/native23_preview_repair_20260913/${taskStamp}_${Case}"
Write-Output "Pinned $Case simulation; full-body teleop remains unqualified."
& wsl -d Ubuntu-22.04 -u root --exec bash -c 'mountpoint -q /mnt/e || mount -t drvfs E: /mnt/e; exec "$@"' -- /mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python $taskScript --case $Case --output $taskOutput
if ($LASTEXITCODE -ne 0) { throw "Simulation failed with exit code $LASTEXITCODE" }
