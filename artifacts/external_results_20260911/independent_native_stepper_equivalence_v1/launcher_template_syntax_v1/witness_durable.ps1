$ErrorActionPreference = 'Stop'
function Get-TaskHash([string] $Path) {
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    try { return ([System.BitConverter]::ToString($algorithm.ComputeHash([System.IO.File]::ReadAllBytes($Path)))).Replace('-', '').ToLowerInvariant() }
    finally { $algorithm.Dispose() }
}
function Write-NewJson([string] $Path, $Value) {
    $bytes = [System.Text.Encoding]::UTF8.GetBytes(($Value | ConvertTo-Json -Depth 16))
    $stream = [System.IO.File]::Open($Path,[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write,[System.IO.FileShare]::Read)
    try { $stream.Write($bytes,0,$bytes.Length) } finally { $stream.Dispose() }
}
$folder = 'E:\codex-artifacts\sonic23_teleop_resume_20260911\independent_native_stepper_equivalence_v1\witness_process'
$receiptPath = Join-Path $folder 'launch_receipt.json'
$receipt = Get-Content -Raw -LiteralPath $receiptPath | ConvertFrom-Json
$clearancePath = Join-Path $folder 'launch_clearance.json'
$clearance = Get-Content -Raw -LiteralPath $clearancePath | ConvertFrom-Json
if (-not $clearance.root_selected_single_run -or $clearance.launch_receipt_sha256 -ne (Get-TaskHash $receiptPath) -or $clearance.request_sha256 -ne $receipt.request_sha256) { throw 'Concrete root-selected launch clearance required.' }
if ((Get-TaskHash $clearance.review.path) -ne $clearance.review.sha256) { throw 'Final review changed.' }
$review = Get-Content -Raw -LiteralPath $clearance.review.path | ConvertFrom-Json
if ($review.($clearance.review.pass_field) -ne $true -or $review.request_subject.sha256 -ne $receipt.request_sha256 -or $review.request_subject.path -ne $receipt.request_path -or $review.launch_receipt_subject.sha256 -ne (Get-TaskHash $receiptPath) -or $review.launch_receipt_subject.path -ne $receiptPath.Replace('\','/')) { throw 'Final review does not name this exact request and launcher.' }
$pre = [ordered]@{}
foreach ($entry in $receipt.input_hashes.PSObject.Properties) {
    $actual = Get-TaskHash $entry.Name
    if ($actual -ne $entry.Value) { throw ('Input changed before launch: '+$entry.Name) }
    $pre[$entry.Name] = [ordered]@{expected=$entry.Value;actual=$actual;matched=$true}
}
$lock = [System.IO.File]::Open((Join-Path $folder 'started.lock'),[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write,[System.IO.FileShare]::None)
$lock.Close()
Write-NewJson (Join-Path $folder 'prerun_hashes.json') ([ordered]@{all_exact=$true;files=$pre})
Write-NewJson (Join-Path $folder 'start.json') ([ordered]@{utc=[DateTime]::UtcNow.ToString('o');wrapper_pid=$PID;receipt_sha256=(Get-TaskHash $receiptPath);clearance_sha256=(Get-TaskHash $clearancePath);review_sha256=(Get-TaskHash $clearance.review.path)})
$childExit = $null
$diagnosticExit = $null
$exitCode = 1
$errorText = $null
try {
    $taskChild = Start-Process -FilePath "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',(Join-Path $folder 'run.ps1')) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $folder 'stdout.log') -RedirectStandardError (Join-Path $folder 'stderr.log')
    $nativeHandle = $taskChild.Handle
    Write-NewJson (Join-Path $folder 'child.json') ([ordered]@{child_pid=$taskChild.Id;wrapper_pid=$PID;handle_acquired=($nativeHandle -ne [IntPtr]::Zero)})
    $taskChild.WaitForExit()
    $childExit = $taskChild.ExitCode
    if ($null -eq $childExit) { throw 'Child exit status unavailable; no automatic retry.' }
    & 'C:/Users/camer/AppData/Local/Programs/Python/Python310/python.exe' 'E:\codex-artifacts\sonic23_teleop_resume_20260911\independent_native_stepper_equivalence_v1\stage_verdict.py' --base 'E:\codex-artifacts\sonic23_teleop_resume_20260911\independent_native_stepper_equivalence_v1' --stage 'witness'
    $diagnosticExit = $LASTEXITCODE
    if ($null -eq $diagnosticExit) { throw 'Diagnostic exit unavailable.' }
    $exitCode = $diagnosticExit
    if ($childExit -ne 0 -and $exitCode -eq 0) { throw 'A nonzero native runner exit cannot pass.' }
} catch { $errorText=$_.Exception.Message; $exitCode=1 } finally {
    $post = [ordered]@{}
    $allExact = $true
    foreach ($entry in $receipt.input_hashes.PSObject.Properties) {
        try { $actual = Get-TaskHash $entry.Name } catch { $actual = $null }
        $post[$entry.Name] = [ordered]@{expected=$entry.Value;actual=$actual;matched=($actual -eq $entry.Value)}
        if ($actual -ne $entry.Value) { $allExact=$false }
    }
    Write-NewJson (Join-Path $folder 'postrun_hashes.json') ([ordered]@{all_exact=$allExact;files=$post})
    if (-not $allExact) { $exitCode=1; if ($null -eq $errorText) { $errorText='Postrun input changed.' } }
    if ((Get-TaskHash $clearancePath) -ne (Get-Content -Raw -LiteralPath (Join-Path $folder 'start.json') | ConvertFrom-Json).clearance_sha256 -or (Get-TaskHash $clearance.review.path) -ne $clearance.review.sha256) { $exitCode=1; $errorText='Final clearance or review changed.' }
    Write-NewJson (Join-Path $folder 'exit.json') ([ordered]@{utc=[DateTime]::UtcNow.ToString('o');exit_code=$exitCode;raw_python_exit_code=$childExit;diagnostic_exit_code=$diagnosticExit;known=($null -ne $childExit);error=$errorText;all_postrun_hashes_exact=$allExact;automatic_retry=$false})
}
exit $exitCode
