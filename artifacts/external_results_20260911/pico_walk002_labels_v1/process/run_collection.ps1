$ErrorActionPreference = 'Stop'
function Get-TaskHash([string] $Path) {
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    try { return ([System.BitConverter]::ToString($algorithm.ComputeHash([System.IO.File]::ReadAllBytes($Path)))).Replace('-', '').ToLowerInvariant() }
    finally { $algorithm.Dispose() }
}
$manifestPath = 'E:\codex-artifacts\sonic23_teleop_resume_20260911\pico_walk002_labels_v1\collector_frozen_inputs.json'
if ((Get-TaskHash $manifestPath) -ne '1aa47048f5c8dfec722995eecc2c68b26d62c343000a86fc15287ff76782c291') { throw 'Frozen manifest changed.' }
$frozen = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
foreach ($entry in $frozen.input_hashes.PSObject.Properties) { if ((Get-TaskHash $entry.Name) -ne $entry.Value) { throw ('Input changed: ' + $entry.Name) } }
$arguments = @(
    '-d',
    'Ubuntu-22.04',
    '--cd',
    '/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof',
    '--',
    'bash',
    '/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh',
    'env',
    'OMP_NUM_THREADS=1',
    'OPENBLAS_NUM_THREADS=1',
    'MKL_NUM_THREADS=1',
    'PYTHONPATH=/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/pico_walk002_labels_v1/source_snapshot_v1',
    '/mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python',
    '/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/pico_walk002_labels_v1/source_snapshot_v1/collect_qualified_rows.py',
    '--manifest',
    '/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/pico_walk002_labels_v1/collector_frozen_inputs.json'
)
& "$env:SystemRoot\System32\wsl.exe" @arguments
$code = $LASTEXITCODE
if ($code -ne 0) { exit $code }
$reportPath = 'E:\codex-artifacts\sonic23_teleop_resume_20260911\pico_walk002_labels_v1\collection\report.json'
$result = Get-Content -Raw -LiteralPath $reportPath | ConvertFrom-Json
if (-not $result.complete -or $result.completed_rows -ne 6847 -or $result.actor_inference_calls -ne 6847 -or $result.physics_steps -ne 0 -or $result.fitting_launched) { throw 'Incomplete or invalid collection result.' }
exit 0
