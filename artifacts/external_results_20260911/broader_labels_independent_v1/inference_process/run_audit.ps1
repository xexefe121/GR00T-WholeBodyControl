$ErrorActionPreference = 'Stop'
function Get-TaskHash([string] $Path) {
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    try { return ([System.BitConverter]::ToString($algorithm.ComputeHash([System.IO.File]::ReadAllBytes($Path)))).Replace('-', '').ToLowerInvariant() }
    finally { $algorithm.Dispose() }
}
$requestPath = 'E:\codex-artifacts\sonic23_teleop_resume_20260911\broader_labels_independent_v1\inference_request_v2.json'
if ((Get-TaskHash $requestPath) -ne 'a45d08b859eb8c6d42d93948911c78e8de7c82fea037ef6a93ca2d8ebf2eae78') { throw 'Audit request changed.' }
$request = Get-Content -Raw -LiteralPath $requestPath | ConvertFrom-Json
foreach ($entry in $request.input_hashes.PSObject.Properties) { if ((Get-TaskHash $entry.Name) -ne $entry.Value) { throw ('Audit input changed: ' + $entry.Name) } }
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
    'PYTHONPATH=/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/broader_labels_independent_v1/source_inference_v2',
    '/mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python',
    '/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/broader_labels_independent_v1/source_inference_v2/audit.py',
    '--request',
    '/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/broader_labels_independent_v1/inference_request_v2.json',
    '--phase',
    'inference'
)
& "$env:SystemRoot\System32\wsl.exe" @arguments
$taskExit = $LASTEXITCODE
if ($null -eq $taskExit) { throw 'WSL exit status unavailable.' }
if ($taskExit -ne 0) { exit $taskExit }
$result = Get-Content -Raw -LiteralPath 'E:\codex-artifacts\sonic23_teleop_resume_20260911\broader_labels_independent_v1\baseline_inference_audit\report.json' | ConvertFrom-Json
if (-not $result.pass_all -or $result.selected_rows_checked -ne 6847 -or $result.actor_inference_calls -ne 6847 -or $result.backward_inference_calls -ne 6847 -or $result.physics_steps -ne 0 -or -not $result.actual_BFM_output_authenticity_verified -or $result.fitting_launched) { throw 'Incomplete independent audit.' }
exit 0
