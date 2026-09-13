$ErrorActionPreference='Stop'
$root=$PSScriptRoot
$checks=[ordered]@{}
foreach($name in @('run_fit_durable_v1.ps1','read_fit_progress_v1.ps1')){
    $tokens=$null;$errors=$null
    [Management.Automation.Language.Parser]::ParseFile((Join-Path $root $name),[ref]$tokens,[ref]$errors)|Out-Null
    if($errors.Count -ne 0){throw "PowerShell AST failed: $name"}
    $checks[$name]=[ordered]@{AST_passed=$true;parse_errors=0}
}
$text=[IO.File]::ReadAllText((Join-Path $root 'run_fit_durable_v1.ps1'))
$guardLine=($text -split "`n" | Where-Object {$_ -match '^\s*if\(\$request.kind'})
if(@($guardLine).Count -ne 1){throw 'Exactly one actual request guard required.'}
$guard=[scriptblock]::Create($guardLine)
$valid='{"kind":"causal_width512_warm_continuation","root_selected":true,"updates":10000,"condition":"causal","fresh_optimizer":false,"ordinary_final_step":81000,"coefficient":1.8188207859141674}'
$request=$valid|ConvertFrom-Json
& $guard
$checks['actual_PS51_decimal_guard_accepts_selected']=$true
$request=($valid.Replace('1.8188207859141674','1.8188207859141675'))|ConvertFrom-Json
$rejected=$false;try{& $guard}catch{$rejected=$true}
if(-not $rejected){throw 'Changed fixed coefficient accepted.'}
$checks['actual_guard_rejects_changed_coefficient']=$true
if($text -notmatch 'WindowStyle Hidden' -or $text -notmatch '\$capturedHandle=\$child.Handle' -or $text -notmatch 'FileShare\]::Delete' -or $text -notmatch 'FileMode\]::CreateNew'){throw 'Durable launch/reader identity contract absent.'}
$checks['durable_hidden_one_shot_and_shared_delete']=$true
$result=[ordered]@{passed=$true;checks=$checks;PowerShell_version=$PSVersionTable.PSVersion.ToString();actual_fit_dispatched=$false;task_model_calls=0;native_steps=0}
$destination=Join-Path $root 'launcher_tests.json'
$stream=[IO.File]::Open($destination,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
try{$writer=[IO.StreamWriter]::new($stream);try{$writer.WriteLine(($result|ConvertTo-Json -Depth 10));$writer.Flush()}finally{$writer.Dispose()}}finally{$stream.Dispose()}
$result|ConvertTo-Json -Depth 10

