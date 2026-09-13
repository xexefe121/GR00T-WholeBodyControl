$ErrorActionPreference = 'Stop'
$base = 'E:/codex-artifacts/sonic23_teleop_resume_20260911/direct_target_fp64_followup_analysis_v1'
$python = 'E:/codex_sonic_runtime/direct_target_gpu_20260911/venv/Scripts/python.exe'
$source = Join-Path $base 'gradient_diagnostic.py'
$request = Join-Path $base 'gradient_request.json'
$clearance = Join-Path $base 'gradient_clearance.json'
$process = Join-Path $base 'gradient_process'
if (Test-Path -LiteralPath $process) { throw 'One process directory already exists; no automatic retry.' }
New-Item -ItemType Directory -Path $process | Out-Null
$lock = [System.IO.File]::Open((Join-Path $process 'started.lock'),[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write,[System.IO.FileShare]::Read)
$lock.Dispose()
$utf8 = [System.Text.UTF8Encoding]::new($false)
function Save-Json([string]$path, $value) {
    [System.IO.File]::WriteAllText($path,($value | ConvertTo-Json -Depth 30),$utf8)
}
function Hash([string]$path) { return (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() }
$rawExit = $null
$runExit = 99
$failure = $null
$child = $null
$childId = $null
try {
    if ((Hash $source) -ne 'c52dec999190b0ffccf226500a0113c02750db43e07f0520b7e3f6392785a8a5') { throw 'Source hash differs.' }
    if ((Hash $request) -ne 'eba680f416038f93c11abcbd0214e0c2f374819fa56fbbb8a79f477f21c07f78') { throw 'Request hash differs.' }
    $pins = Get-Content -Raw -LiteralPath $request | ConvertFrom-Json
    $gate = Get-Content -Raw -LiteralPath $clearance | ConvertFrom-Json
    if (-not $gate.approved -or $gate.request_sha256 -ne (Hash $request) -or (Hash $gate.review_path) -ne $gate.review_sha256) { throw 'Concrete review binding differs.' }
    if (Test-Path -LiteralPath (Join-Path $base 'gradient_results')) { throw 'Diagnostic outputs already exist.' }
    $env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
    $env:OMP_NUM_THREADS = '1'
    $env:OPENBLAS_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    $env:PYTHONUTF8 = '1'
    $env:PYTHONPATH = $pins.original_source_directory
    $arguments = @('-u',('"' + $source + '"'))
    $child = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $base -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $process 'stdout.log') -RedirectStandardError (Join-Path $process 'stderr.log')
    $capturedHandle = $child.Handle
    $childId = $child.Id
    Save-Json (Join-Path $process 'launch.json') @{ started_utc=[DateTime]::UtcNow.ToString('o'); wrapper_pid=$PID; child_pid=$childId; executable=$python; arguments=$arguments; working_directory=$base; source_sha256=(Hash $source); request_sha256=(Hash $request); clearance_sha256=(Hash $clearance); launcher_sha256=(Hash $PSCommandPath); captured_child_handle=$true }
    $child.WaitForExit()
    $rawExit = $child.ExitCode
    if ($null -eq $rawExit) { throw 'Child exit status is unknown.' }
    $runExit = [int]$rawExit
    $post = @{}
    foreach ($pin in $pins.input_sha256.PSObject.Properties) {
        $actual = Hash $pin.Name
        $post[$pin.Name] = $actual
        if ($actual -ne $pin.Value) { throw ('Changed final input: ' + $pin.Name) }
    }
    foreach ($path in @($source,$request,$clearance,$PSCommandPath)) { $post[$path] = Hash $path }
    Save-Json (Join-Path $process 'postrun_sha256.json') $post
    if ($runExit -eq 0) {
        $report = Get-Content -Raw -LiteralPath (Join-Path $base 'gradient_results/report.json') | ConvertFrom-Json
        if (-not $report.passed -or $report.request_sha256 -ne (Hash $request)) { $runExit = 1 }
    }
} catch {
    $failure = $_.Exception.ToString()
    if ($runExit -eq 0 -or $null -eq $runExit) { $runExit = 99 }
} finally {
    Save-Json (Join-Path $process 'exit.json') @{ ended_utc=[DateTime]::UtcNow.ToString('o'); wrapper_pid=$PID; child_pid=$childId; raw_python_exit_code=$rawExit; raw_exit_known=($null -ne $rawExit); exit_code=$runExit; wrapper_error=$failure; no_automatic_retry=$true }
    if ($null -ne $child) { $child.Dispose() }
}
exit $runExit
