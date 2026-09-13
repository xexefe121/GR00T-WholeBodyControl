$ErrorActionPreference='Stop'
$base=$PSScriptRoot
$path=Join-Path $base 'template_preview.ps1.txt'
$tokens=$null;$errors=$null
$null=[System.Management.Automation.Language.Parser]::ParseFile($path,[ref]$tokens,[ref]$errors)
if($errors.Count -ne 0){throw 'Generated saved audit PowerShell parse failed.'}
$receiptPath=Join-Path $base 'launch_receipt.json'
$normal=$receiptPath.Replace([char]92,[char]47)
$expected=($base.Replace([char]92,[char]47)+'/launch_receipt.json')
if($normal -cne $expected){throw 'Actual Windows path normalization failed.'}
$hash=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
$value=[ordered]@{passed=$true;synthetic_only=$true;windows_path_normalization_passed=$true;source_sha256=@{'template_preview.ps1.txt'=$hash};generated_scripts_executed=$false;native_steps=0;model_calls=0;worker_processes=0}
$target=Join-Path $base 'template_parse.json'
$stream=[IO.File]::Open($target,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
try{$bytes=[Text.Encoding]::UTF8.GetBytes(($value|ConvertTo-Json -Depth 20));$stream.Write($bytes,0,$bytes.Length)}finally{$stream.Dispose()}
