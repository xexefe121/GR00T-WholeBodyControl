$ErrorActionPreference='Stop'
$base=$PSScriptRoot
$files=@('preview_run.ps1.txt','preview_durable.ps1.txt')
$hashes=[ordered]@{}
foreach($name in $files){
  $path=Join-Path $base $name
  $tokens=$null; $errors=$null
  $null=[System.Management.Automation.Language.Parser]::ParseFile($path,[ref]$tokens,[ref]$errors)
  if($errors.Count -ne 0){throw 'Generated PowerShell parse failed.'}
  $hashes[$name]=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
}
$receiptPath=Join-Path (Join-Path $base 'clock_process') 'launch_receipt.json'
$normalized=$receiptPath.Replace('\','/')
$expected=($base.Replace('\','/')+'/clock_process/launch_receipt.json')
if($normalized -cne $expected){throw 'Actual Windows path normalization failed.'}
$value=[ordered]@{passed=$true;synthetic_only=$true;windows_path_normalization_passed=$true;source_sha256=$hashes;generated_scripts_executed=$false;native_steps=0;model_calls=0;worker_processes=0}
$target=Join-Path $base 'template_parse.json'
$stream=[IO.File]::Open($target,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
try{$bytes=[Text.Encoding]::UTF8.GetBytes(($value|ConvertTo-Json -Depth 20));$stream.Write($bytes,0,$bytes.Length)}finally{$stream.Dispose()}
