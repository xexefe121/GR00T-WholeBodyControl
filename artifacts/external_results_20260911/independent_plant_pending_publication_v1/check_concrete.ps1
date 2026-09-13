$ErrorActionPreference='Stop'
$items=@()
foreach($name in @('run.ps1','run_durable.ps1')){
  $path=Join-Path (Join-Path $PSScriptRoot 'clock_process') $name
  $tokens=$null;$errors=$null
  $null=[Management.Automation.Language.Parser]::ParseFile($path,[ref]$tokens,[ref]$errors)
  if($errors.Count -ne 0){throw 'Actual concrete script did not parse.'}
  $items+=@{path=$path.Replace('\','/');passed=$true;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}
}
$r=Join-Path (Join-Path $PSScriptRoot 'clock_process') 'launch_receipt.json'
$normalized=$r.Replace('\','/')
$expected=$PSScriptRoot.Replace('\','/')+'/clock_process/launch_receipt.json'
if($normalized -cne $expected){throw 'Concrete Windows separator normalization failed.'}
$value=@{passed=$true;normalization_passed=$true;literal_separator_length='\'.Length;literal_separator_codepoint=[int][char]'\';launchers=$items;scripts_executed=$false}
$target=Join-Path $PSScriptRoot 'concrete_launcher_parse.json'
$s=[IO.File]::Open($target,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
try{$b=[Text.Encoding]::UTF8.GetBytes(($value|ConvertTo-Json -Depth 20));$s.Write($b,0,$b.Length)}finally{$s.Dispose()}
