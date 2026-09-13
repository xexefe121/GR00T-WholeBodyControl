$ErrorActionPreference='Stop'
$base=$PSScriptRoot
if($PSVersionTable.PSVersion.Major -ne 5){throw 'Test requires actual selected Windows PowerShell 5.1'}
$checks=[ordered]@{}
foreach($name in @('run_fit_durable_v1.ps1','dispatch_fit.ps1')){
    $tokens=$null;$errors=$null;$ast=[Management.Automation.Language.Parser]::ParseFile((Join-Path $base $name),[ref]$tokens,[ref]$errors)
    if($errors.Count -ne 0){throw 'PowerShell AST rejected'}
    $checks[($name+'_AST')]=$true
}
$source=[IO.File]::ReadAllText((Join-Path $base 'run_fit_durable_v1.ps1'))
$line=($source -split "`n" | Where-Object {$_ -like "*throw 'Unexpected request.'*"})
if(@($line).Count -ne 1){throw 'One literal selected request guard required'}
$proof=Get-Content -LiteralPath (Join-Path $base 'metadata_schema_proof.json') -Raw|ConvertFrom-Json
$request=$proof.request_candidate
$guard=[ScriptBlock]::Create($line)
& $guard
$checks['actual_request_guard_accepts']=$true
$request.coefficient=[decimal]1.8188207859141675d
$rejected=$false;try{& $guard}catch{$rejected=$true}
if(-not $rejected){throw 'Changed decimal coefficient accepted'}
$checks['last_decimal_coefficient_rejected']=$true
$request=$proof.request_candidate;$request.coefficient=[decimal]1.8188207859141674d;$request.recovery_coefficient=[decimal]0.21d
$rejected=$false;try{& $guard}catch{$rejected=$true}
if(-not $rejected){throw 'Changed recovery coefficient accepted'}
$checks['changed_D3_weight_rejected']=$true
foreach($literal in @('LaunchReceiptSha256','ClearanceSha256','-WindowStyle Hidden','$capturedHandle=$child.Handle','$child.WaitForExit()','[IO.FileMode]::CreateNew','[IO.FileShare]::Delete','launch_receipt_sha256=$LaunchReceiptSha256')){
    if(-not $source.Contains($literal)){throw ('Missing preserved wrapper behavior: '+$literal)}
    $checks[$literal]=$true
}
if($source.IndexOf('$capturedHandle=$child.Handle') -gt $source.IndexOf('$child.WaitForExit()')){throw 'Handle captured after wait'}
$checks['handle_before_wait']=$true
$result=[ordered]@{passed=$true;checks=$checks;count=$checks.Count;PSVersion=$PSVersionTable.PSVersion.ToString();actual_task_dispatches=0;model_calls=0;native_steps=0}
$path=Join-Path $base 'launcher_tests.json'
$stream=[IO.File]::Open($path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
try{$writer=[IO.StreamWriter]::new($stream,[Text.UTF8Encoding]::new($false));try{$writer.Write(($result|ConvertTo-Json -Depth 10));$writer.WriteLine()}finally{$writer.Dispose()}}finally{$stream.Dispose()}
$result|ConvertTo-Json -Depth 3
