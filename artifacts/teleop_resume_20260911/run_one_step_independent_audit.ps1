param(
    [Parameter(Mandatory=$true)][ValidateSet('physics','labels')][string]$Audit,
    [Parameter(Mandatory=$true)][ValidatePattern('^[a-f0-9]{64}$')][string]$CollectionReportSha256,
    [ValidatePattern('^[a-f0-9]{64}$')][string]$PhysicsReportSha256
)
$ErrorActionPreference = 'Stop'
$taskRoot = 'Z:\codex\GR00T-WholeBodyControl-sonic-transfer-23dof'
$taskArtifacts = Join-Path $taskRoot 'artifacts\teleop_resume_20260911'
$taskData = 'E:\codex-artifacts\sonic23_teleop_resume_20260911'
$taskWslData = '/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911'
$taskOutput = Join-Path $taskData ('one_step_independent_' + $Audit + '_v1')
$taskProcess = Join-Path $taskData ('one_step_independent_' + $Audit + '_process_v1')
if ((Test-Path -LiteralPath $taskOutput) -or (Test-Path -LiteralPath $taskProcess)) {
    throw 'Existing audit attempt must be preserved; this launcher never resumes or reruns.'
}
New-Item -ItemType Directory -Path $taskProcess -ErrorAction Stop | Out-Null
function Read-TaskHash([string]$Path) {
    $algorithm = [Security.Cryptography.SHA256]::Create()
    $stream = [IO.File]::Open($Path,[IO.FileMode]::Open,[IO.FileAccess]::Read,([IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete))
    try { return ([BitConverter]::ToString($algorithm.ComputeHash($stream))).Replace('-','').ToLowerInvariant() }
    finally { $stream.Dispose(); $algorithm.Dispose() }
}
function Read-TaskJson([string]$Path) {
    $stream = [IO.File]::Open($Path,[IO.FileMode]::Open,[IO.FileAccess]::Read,([IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete))
    try { $reader = New-Object IO.StreamReader($stream); try { return ($reader.ReadToEnd() | ConvertFrom-Json) } finally { $reader.Dispose() } }
    finally { $stream.Dispose() }
}
$taskExit = 1; $taskRawExit = $null; $taskError = $null
try {
    $taskPins = [ordered]@{}
    $taskReview = Join-Path $taskData 'one_step_independent_audits_source_review_v1\review.json'
    if ((Read-TaskHash $taskReview) -ne '4670259e3186a136db5c8618bb9eade4b3d5cd59d4d75d9f87ece789f5de7bf3') { throw 'Audit source review changed.' }
    $taskPins[$taskReview] = Read-TaskHash $taskReview
    $taskBootstrap = Join-Path $taskRoot 'artifacts\teleop_six_hour_20260910\RUN_PASSING_WALK_WSL.sh'
    $taskBootstrapHash = '392de6eccb281c41219566c4b7c9c1813b086913f66d861529d4904959168ebc'
    if ((Read-TaskHash $taskBootstrap) -ne $taskBootstrapHash) { throw 'WSL bootstrap changed.' }
    $taskPins[$taskBootstrap] = $taskBootstrapHash
    $review = Read-TaskJson $taskReview
    if ($review.source_review_pass -ne $true) { throw 'Audit source review did not pass.' }
    foreach ($entry in $review.source_sha256.PSObject.Properties) {
        if ((Read-TaskHash $entry.Name) -ne $entry.Value) { throw ('Audit source changed: ' + $entry.Name) }
        $taskPins[$entry.Name] = $entry.Value
    }
    $taskCollection = Join-Path $taskData 'one_step_policy_branch_collection_resume2969_v1\collection'
    $taskCollectionReport = Join-Path $taskCollection 'report.json'
    if ((Read-TaskHash $taskCollectionReport) -ne $CollectionReportSha256) { throw 'Collection report changed.' }
    $producer = Read-TaskJson $taskCollectionReport
    if ($producer.completed -ne $true -or $producer.passed -ne $true -or $producer.rows -ne 3054) { throw 'Collection incomplete.' }
    $taskPins[$taskCollectionReport] = $CollectionReportSha256
    $taskPins[$PSCommandPath] = Read-TaskHash $PSCommandPath
    $taskArgs = @('-d','Ubuntu-22.04','--cd','/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof',
        '--','bash','/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh',
        'env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',
        '/mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python',
        ('artifacts/teleop_resume_20260911/audit_one_step_' + $Audit + '.py'),
        '--collection',($taskWslData + '/one_step_policy_branch_collection_resume2969_v1/collection'),
        '--collection-report-sha256',$CollectionReportSha256,
        '--output',($taskWslData + '/one_step_independent_' + $Audit + '_v1'))
    if ($Audit -eq 'physics') {
        $taskArgs += @('--old-snapshots',($taskWslData + '/old_expert_prefix_snapshots_v2/capture/control_snapshots.npz'),
            '--old-report-sha256','fb0226872958f414b9fdf263c54ca340733a697b7fea82c86215f872cbd28e30')
    } else {
        if (-not $PhysicsReportSha256) { throw 'Labels require the completed physics report hash.' }
        $taskPhysicsReport = Join-Path $taskData 'one_step_independent_physics_v1\report.json'
        if ((Read-TaskHash $taskPhysicsReport) -ne $PhysicsReportSha256) { throw 'Physics report changed.' }
        if ((Read-TaskJson $taskPhysicsReport).passed -ne $true) { throw 'Physics audit did not pass.' }
        $taskPins[$taskPhysicsReport] = $PhysicsReportSha256
        $taskArgs += @('--physics-audit',($taskWslData + '/one_step_independent_physics_v1'),
            '--physics-report-sha256',$PhysicsReportSha256)
    }
    [ordered]@{utc=[DateTime]::UtcNow.ToString('o');wrapper_pid=$PID;audit=$Audit;arguments=$taskArgs;
        input_sha256=$taskPins;maximum_native_steps=$(if ($Audit -eq 'physics') {61080} else {0});graph_calls=0;optimizer_updates=0} |
        ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $taskProcess 'launch.json') -Encoding UTF8
    $ErrorActionPreference = 'Continue'
    & wsl.exe @taskArgs 1> (Join-Path $taskProcess 'stdout.log') 2> (Join-Path $taskProcess 'stderr.log')
    $taskRawExit = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($null -eq $taskRawExit) { throw 'WSL exit status unavailable.' }
    $taskExit = $taskRawExit
    $taskPost = [ordered]@{}
    foreach ($entry in $taskPins.GetEnumerator()) {
        $taskPost[$entry.Key] = Read-TaskHash $entry.Key
        if ($taskPost[$entry.Key] -ne $entry.Value) { throw ('Postrun input changed: ' + $entry.Key) }
    }
    $taskPost | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $taskProcess 'postrun_hashes.json') -Encoding UTF8
    if ($taskRawExit -eq 0 -and (Read-TaskJson (Join-Path $taskOutput 'report.json')).passed -ne $true) { throw 'Missing passed audit report.' }
} catch { $taskExit=1; $taskError=$_.Exception.Message }
finally {
    [ordered]@{utc=[DateTime]::UtcNow.ToString('o');wrapper_pid=$PID;raw_wsl_exit_code=$taskRawExit;exit_code=$taskExit;error=$taskError} |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskProcess 'exit.json') -Encoding UTF8
}
exit $taskExit
