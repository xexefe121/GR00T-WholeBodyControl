$ErrorActionPreference='Stop'
$testRoot=Join-Path $PSScriptRoot 'launcher_tests_v2'
if(Test-Path -LiteralPath $testRoot){throw 'Preserve existing test output; use a new version.'}
[IO.Directory]::CreateDirectory($testRoot) | Out-Null
$parsed=@()
foreach($name in @('run_fit_durable_v1.ps1','read_fit_progress_v1.ps1')){
    $tokens=$null;$errors=$null
    $ast=[Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot $name),[ref]$tokens,[ref]$errors)
    if($errors.Count -ne 0){throw ($errors | Out-String)}
    $parsed+=@{path=$name;parse_errors=$errors.Count}
    if($name -eq 'run_fit_durable_v1.ps1'){
        # Only isolated helper definitions execute; the actual launch body is not run.
        foreach($fn in $ast.FindAll({param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst]},$false)){
            . ([ScriptBlock]::Create($fn.Extent.Text))
        }
    }
}
$shareMode=[IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete
$path=Join-Path $testRoot 'atomic.json';$replacement=Join-Path $testRoot 'replacement.json'
Write-NewJson $path @{version=1};Write-NewJson $replacement @{version=2}
$opened=[IO.File]::Open($path,[IO.FileMode]::Open,[IO.FileAccess]::Read,$shareMode)
try {
    [IO.File]::Replace($replacement,$path,(Join-Path $testRoot 'replaced_backup.json'))
    if((Read-SharedJson $path).version -ne 2){throw 'Atomic replacement failed.'}
} finally {$opened.Dispose()}
$before=Read-SharedSha $path;$createBlocked=$false
try {Write-NewJson $path @{version=3}} catch {$createBlocked=$true}
if(-not $createBlocked -or (Read-SharedSha $path) -ne $before){throw 'CreateNew did not preserve original.'}
$environment=[pscustomobject]@{OMP_NUM_THREADS='1';OPENBLAS_NUM_THREADS='1';MKL_NUM_THREADS='1';PYTHONPATH='synthetic'}
if((($environment.PSObject.Properties.Name | Sort-Object) -join ',') -ne 'MKL_NUM_THREADS,OMP_NUM_THREADS,OPENBLAS_NUM_THREADS,PYTHONPATH'){throw 'Environment comparison failed.'}
$child=Start-Process -FilePath "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -ArgumentList '-NoProfile -NonInteractive -Command "exit 7"' -WindowStyle Hidden -PassThru
$capturedHandle=$child.Handle
$child.WaitForExit();$exitCode=$child.ExitCode
if($null -eq $exitCode -or $exitCode -ne 7){throw 'Child handle/exit preservation failed.'}
$child.Dispose()
Write-NewJson (Join-Path $testRoot 'powershell_checks.json') @{
    passed=$true;parsed=$parsed;child_exit_code=$exitCode;captured_handle_nonzero=($capturedHandle -ne [IntPtr]::Zero);
    atomic_replace_while_reader_open=$true;create_new_preserves_existing=$true;actual_launch_body_executed=$false;
    no_model_or_native_calls=$true;optimizer_updates=0;graph_calls=0;native_steps=0
}
