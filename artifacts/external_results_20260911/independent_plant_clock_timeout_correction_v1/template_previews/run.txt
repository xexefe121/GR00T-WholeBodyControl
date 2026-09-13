$ErrorActionPreference = 'Stop'
function Get-TaskHash([string] $Path) {
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    $stream = [System.IO.File]::Open($Path,[System.IO.FileMode]::Open,[System.IO.FileAccess]::Read,([System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete))
    try { return ([System.BitConverter]::ToString($algorithm.ComputeHash($stream))).Replace('-', '').ToLowerInvariant() }
    finally { $stream.Dispose(); $algorithm.Dispose() }
}
function Write-NewJson([string] $Path, $Value) {
    $bytes = [System.Text.Encoding]::UTF8.GetBytes(($Value | ConvertTo-Json -Depth 16))
    $stream = [System.IO.File]::Open($Path,[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write,[System.IO.FileShare]::Read)
    try { $stream.Write($bytes,0,$bytes.Length) } finally { $stream.Dispose() }
}
$folder = 'E:\codex-artifacts\sonic23_teleop_resume_20260911\independent_plant_clock_timeout_correction_v1\clock_process'
$arguments = @(
'-d',
'Ubuntu-22.04',
'--cd',
'/',
'--',
'bash',
'/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh',
'timeout',
'--signal=TERM',
'--kill-after=5s',
'555s',
'env',
'OMP_NUM_THREADS=1',
'OPENBLAS_NUM_THREADS=1',
'MKL_NUM_THREADS=1',
'NUMEXPR_NUM_THREADS=1',
'PYTHONDONTWRITEBYTECODE=1',
'PYTHONPATH=/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/independent_plant_clock_timeout_correction_v1/source_draft_v1',
'/mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python',
'-B',
'/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/independent_plant_clock_timeout_correction_v1/source_draft_v1/run_clock.py',
'--request',
'/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/independent_plant_clock_timeout_correction_v1/clock_request.json',
'--clearance',
'/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/independent_plant_clock_timeout_correction_v1/clock_process/launch_clearance.json'
)
$rawExit = $null
$rawError = $null
try {
    & "$env:SystemRoot\System32\wsl.exe" @arguments
    $rawExit = $LASTEXITCODE
} catch { $rawError = $_.Exception.Message } finally {
    Write-NewJson (Join-Path $folder 'raw_exit.json') ([ordered]@{utc=[DateTime]::UtcNow.ToString('o');raw_python_exit_code=$rawExit;raw_error=$rawError;known=($null -ne $rawExit)})
}
if ($null -eq $rawExit) { exit 1 }
exit $rawExit
