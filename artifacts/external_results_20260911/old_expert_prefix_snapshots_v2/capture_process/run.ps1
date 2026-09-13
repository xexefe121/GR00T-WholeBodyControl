$ErrorActionPreference = 'Stop'
function Get-TaskHash([string] $Path) {
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    try { return ([System.BitConverter]::ToString($algorithm.ComputeHash([System.IO.File]::ReadAllBytes($Path)))).Replace('-', '').ToLowerInvariant() }
    finally { $algorithm.Dispose() }
}
$requestPath = 'E:\codex-artifacts\sonic23_teleop_resume_20260911\old_expert_prefix_snapshots_v2\capture_request.json'
if ((Get-TaskHash $requestPath) -ne '415905ad1e5ea5938ff0bda82155342caaae7ed13b72ed952832ff65b69a6c9a') { throw 'Capture request changed.' }
$request = Get-Content -Raw -LiteralPath $requestPath | ConvertFrom-Json
foreach ($entry in $request.input_hashes.PSObject.Properties) { if ((Get-TaskHash $entry.Name) -ne $entry.Value) { throw ('Capture input changed: ' + $entry.Name) } }
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
    'PYTHONPATH=/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/old_expert_prefix_snapshots_v2/source_snapshot_v1',
    '/mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python',
    '/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/old_expert_prefix_snapshots_v2/source_snapshot_v1/capture_old_prefix.py'
)
& "$env:SystemRoot\System32\wsl.exe" @arguments
$taskExit = $LASTEXITCODE
if ($null -eq $taskExit) { throw 'WSL exit status unavailable.' }
if ($taskExit -ne 0) { exit $taskExit }
$result = Get-Content -Raw -LiteralPath 'E:\codex-artifacts\sonic23_teleop_resume_20260911\old_expert_prefix_snapshots_v2\capture\report.json' | ConvertFrom-Json
if (-not $result.passed -or $result.completed_controls -ne 1268 -or $result.completed_native_steps -ne 12680 -or $result.verified_native_steps -ne 12680 -or $result.snapshots -ne 1269 -or $result.new_inference_calls -ne 0 -or $result.new_labels -ne 0 -or $result.new_perturbations -ne 0 -or $result.original_full_lifecycle_reexecuted -or $result.last_snapshot_control_executed) { throw 'Incomplete or out-of-scope prefix capture.' }
exit 0
