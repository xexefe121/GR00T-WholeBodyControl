$ErrorActionPreference = 'Stop'
function Get-TaskHash([string] $Path) {
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    try { return ([System.BitConverter]::ToString($algorithm.ComputeHash([System.IO.File]::ReadAllBytes($Path)))).Replace('-', '').ToLowerInvariant() }
    finally { $algorithm.Dispose() }
}
$bindingPath = 'Y:\Temp\tmpr9psa_ct\evaluation_binding.json'
if ((Get-TaskHash $bindingPath) -ne 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa') { throw 'Final binding changed.' }
$binding = Get-Content -Raw -LiteralPath $bindingPath | ConvertFrom-Json
foreach ($entry in $binding.input_files) { if ((Get-TaskHash $entry.path) -ne $entry.sha256) { throw ('Bound input changed: ' + $entry.path) } }
if (Test-Path -LiteralPath 'Y:\Temp\tmpr9psa_ct\nominal') { throw 'Existing attempt output must be preserved.' }
if (Test-Path -LiteralPath 'Y:\Temp\tmpr9psa_ct\post_lifecycle_hold_5s') { throw 'Existing attempt output must be preserved.' }
if (Test-Path -LiteralPath 'Y:\Temp\tmpr9psa_ct\pilot_outcome.json') { throw 'Existing attempt output must be preserved.' }
$executionLock = [System.IO.File]::Open('Y:\Temp\tmpr9psa_ct\evaluation_process\execution.lock',[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write,[System.IO.FileShare]::None)
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
    'PYTHONPATH=/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/tmpr9psa_ct/source_snapshot_v1',
    '/mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python',
    '/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/tmpr9psa_ct/source_snapshot_v1/evaluate_physical_response_student.py'
)
& "$env:SystemRoot\System32\wsl.exe" @arguments
$taskExit = $LASTEXITCODE
if ($null -eq $taskExit) { throw 'WSL exit status unavailable.' }
if ($taskExit -ne 0) { exit $taskExit }
$result = Get-Content -Raw -LiteralPath 'Y:\Temp\tmpr9psa_ct\pilot_outcome.json' | ConvertFrom-Json
if (-not $result.nominal.full_segment_completed -or -not $result.extension.full_segment_completed -or $result.nominal.completed_controls -ne 1569 -or $result.extension.completed_controls -ne 250 -or $result.nominal.physics_steps -ne 15690 -or $result.extension.physics_steps -ne 2500) { throw 'Canonical evaluation stopped or incomplete; preserve first failure.' }
if (-not $result.nominal.quiet_standing_diagnostic.quiet_standing_diagnostic_pass -or -not $result.extension.quiet_standing_diagnostic.quiet_standing_diagnostic_pass) { throw 'Physical completion retained; quiet standing failed.' }
$parity = Get-Content -Raw -LiteralPath 'Y:\Temp\tmpr9psa_ct\canonical_prefix250_parity.json' | ConvertFrom-Json
if (-not $parity.passed) { throw 'Canonical transition parity failed.' }
$parity = Get-Content -Raw -LiteralPath 'Y:\Temp\tmpr9psa_ct\actual_query250_input_parity.json' | ConvertFrom-Json
if (-not $parity.passed) { throw 'Canonical transition parity failed.' }
$parity = Get-Content -Raw -LiteralPath 'Y:\Temp\tmpr9psa_ct\actual_query250_ownexport_output_parity.json' | ConvertFrom-Json
if (-not $parity.passed) { throw 'Canonical transition parity failed.' }
exit 0
