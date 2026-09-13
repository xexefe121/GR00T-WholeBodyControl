$ErrorActionPreference = 'Stop'
function Get-TaskHash([string] $Path) {
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    try { return ([System.BitConverter]::ToString($algorithm.ComputeHash([System.IO.File]::ReadAllBytes($Path)))).Replace('-', '').ToLowerInvariant() }
    finally { $algorithm.Dispose() }
}
$bindingPath = 'Y:\Temp\tmpnnibqjk5\witness_binding.json'
if ((Get-TaskHash $bindingPath) -ne 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa') { throw 'Final binding changed.' }
$binding = Get-Content -Raw -LiteralPath $bindingPath | ConvertFrom-Json
foreach ($entry in $binding.input_files) { if ((Get-TaskHash $entry.path) -ne $entry.sha256) { throw ('Bound input changed: ' + $entry.path) } }
if (Test-Path -LiteralPath 'Y:\Temp\tmpnnibqjk5\head_witness') { throw 'Existing attempt output must be preserved.' }
$executionLock = [System.IO.File]::Open('Y:\Temp\tmpnnibqjk5\witness_process\execution.lock',[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write,[System.IO.FileShare]::None)
$executionLock.Close()
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
    'PYTHONPATH=/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/tmpnnibqjk5/source_snapshot_v1',
    '/mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python',
    '/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/tmpnnibqjk5/source_snapshot_v1/head_activation_witness.py'
)
& "$env:SystemRoot\System32\wsl.exe" @arguments
$taskExit = $LASTEXITCODE
if ($null -eq $taskExit) { throw 'WSL exit status unavailable.' }
if ($taskExit -ne 0) { exit $taskExit }
$result = Get-Content -Raw -LiteralPath 'Y:\Temp\tmpnnibqjk5\head_witness\report.json' | ConvertFrom-Json
if (-not $result.pass_all -or $result.attempted_head_calls -ne 1 -or $result.expected_head_calls -ne 1 -or $result.BFM_inference_calls -ne 0 -or $result.physics_steps -ne 0 -or $result.fitting_launched) { throw 'Single-call witness incomplete.' }
exit 0
