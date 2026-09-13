$ErrorActionPreference = 'Stop'
function Get-TaskHash([string] $Path) {
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    $stream = [System.IO.File]::Open($Path,[System.IO.FileMode]::Open,[System.IO.FileAccess]::Read,([System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete))
    try { return ([System.BitConverter]::ToString($algorithm.ComputeHash($stream))).Replace('-', '').ToLowerInvariant() }
    finally { $stream.Dispose(); $algorithm.Dispose() }
}
function Write-NewJson([string] $Path, $Value) {
    $text = $Value | ConvertTo-Json -Depth 12
    $stream = [System.IO.File]::Open($Path,[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write,[System.IO.FileShare]::Read)
    try { $bytes = [System.Text.Encoding]::UTF8.GetBytes($text); $stream.Write($bytes,0,$bytes.Length) } finally { $stream.Dispose() }
}
$folder = 'E:\codex-artifacts\sonic23_teleop_resume_20260911\direct_target_causal_response_evaluation_v1\witness_process'
$bindingPath = 'E:\codex-artifacts\sonic23_teleop_resume_20260911\direct_target_causal_response_evaluation_v1\witness_binding.json'
if ((Get-TaskHash $bindingPath) -ne '2a56cf0ab008f866625a336c8d6445fec2cebcfb4d55f26e8cf3995c0fa33f2c') { throw 'Final binding changed.' }
$binding = Get-Content -Raw -LiteralPath $bindingPath | ConvertFrom-Json
foreach ($entry in $binding.input_files) { if ((Get-TaskHash $entry.path) -ne $entry.sha256) { throw ('Bound input changed: ' + $entry.path) } }
$arguments = @(
    '-d',
    'Ubuntu-22.04',
    '--cd',
    '/',
    '--',
    'bash',
    '/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh',
    'env',
    'OMP_NUM_THREADS=1',
    'OPENBLAS_NUM_THREADS=1',
    'MKL_NUM_THREADS=1',
    'PYTHONDONTWRITEBYTECODE=1',
    'PYTHONPATH=/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/direct_target_causal_response_evaluation_v1/source_draft_v1',
    '/mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python',
    '-B',
    '/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/direct_target_causal_response_evaluation_v1/source_draft_v1/head_activation_witness.py'
)
$rawExit = $null
$rawError = $null
try {
    & "$env:SystemRoot\System32\wsl.exe" @arguments
    $rawExit = $LASTEXITCODE
} catch { $rawError = $_.Exception.Message } finally {
    Write-NewJson (Join-Path $folder 'raw_exit.json') ([ordered]@{utc=[DateTime]::UtcNow.ToString('o');raw_python_exit_code=$rawExit;raw_error=$rawError;known=($null -ne $rawExit)})
}
& 'C:/Users/camer/AppData/Local/Programs/Python/Python310/python.exe' 'E:\codex-artifacts\sonic23_teleop_resume_20260911\direct_target_causal_response_evaluation_v1\diagnostic_verdict.py' --base 'E:\codex-artifacts\sonic23_teleop_resume_20260911\direct_target_causal_response_evaluation_v1' --mode 'witness' --raw-record (Join-Path $folder 'raw_exit.json') --output (Join-Path $folder 'diagnostic_verdict.json')
$verdictExit = $LASTEXITCODE
if ($null -eq $verdictExit) { throw 'Diagnostic exit unavailable.' }
exit $verdictExit
