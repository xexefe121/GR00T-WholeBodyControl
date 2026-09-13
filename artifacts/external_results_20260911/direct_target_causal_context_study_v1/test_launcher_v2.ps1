$ErrorActionPreference='Stop'
$taskRoot=$PSScriptRoot
$request=Get-Content -LiteralPath (Join-Path $taskRoot 'training_request.json') -Raw|ConvertFrom-Json
$source=Get-Content -LiteralPath (Join-Path $taskRoot 'run_fit_durable_v2.ps1') -Raw
$line=($source -split "`n" | Where-Object {$_ -match "throw 'Unexpected request'"})
# Match exact actual guard, including its terminal full stop.
if(-not $line){$line=($source -split "`n" | Where-Object {$_ -match "Unexpected request\."})}
if(@($line).Count -ne 1){throw 'Actual guard not unique'}
$guard=[ScriptBlock]::Create($line)
& $guard
$originalUnequal=($request.coefficient -ne 1.8188207859141674)
$decimalEqual=($request.coefficient -eq 1.8188207859141674d)
if(-not $originalUnequal -or -not $decimalEqual){throw 'Expected pinned PS5.1 numeric behavior differs'}
$old=$request.coefficient;$request.coefficient=1.8188207859141675d;$rejected=$false
try{& $guard}catch{$rejected=$true}
$request.coefficient=$old
if(-not $rejected){throw 'Changed coefficient was accepted'}
$asts=@()
foreach($name in @('run_fit_durable_v2.ps1','read_fit_progress_v2.ps1')){
    $tokens=$null;$errors=$null
    $ast=[System.Management.Automation.Language.Parser]::ParseFile((Join-Path $taskRoot $name),[ref]$tokens,[ref]$errors)
    if($errors.Count){throw 'PowerShell parse failed'}
    $asts+=@{file=$name;passed=$true}
}
$result=[ordered]@{passed=$true;PSVersion=$PSVersionTable.PSVersion.ToString();json_coefficient_type=$request.coefficient.GetType().FullName;
    original_mixed_comparison_unequal=$originalUnequal;decimal_comparison_equal=$decimalEqual;actual_corrected_guard_passed=$true;
    changed_coefficient_rejected=$rejected;asts=$asts;task_model_calls=0;optimizer_updates=0;native_calls=0}
$output=Join-Path $taskRoot 'launcher_v2_test.json'
if(Test-Path -LiteralPath $output){throw 'Prior test evidence exists'}
$result|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $output -Encoding UTF8
$result|ConvertTo-Json -Depth 8
