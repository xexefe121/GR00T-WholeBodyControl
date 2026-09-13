$ErrorActionPreference='Stop'
$folder='E:\codex-artifacts\sonic23_teleop_resume_20260911\direct_target_causal_width512_evaluation_v1\witness_process'
$files=[ordered]@{}
foreach($name in @('run.ps1','run_durable.ps1')) {
    $path=Join-Path $folder $name
    $tokens=$null
    $parseErrors=$null
    $null=[System.Management.Automation.Language.Parser]::ParseFile($path,[ref]$tokens,[ref]$parseErrors)
    if($parseErrors.Count -ne 0){throw ($parseErrors | Out-String)}
    $algorithm=[System.Security.Cryptography.SHA256]::Create()
    $stream=[System.IO.File]::OpenRead($path)
    try {$files[$name]=([System.BitConverter]::ToString($algorithm.ComputeHash($stream))).Replace('-','').ToLowerInvariant()}
    finally {$stream.Dispose();$algorithm.Dispose()}
}
$destination=Join-Path $PSScriptRoot 'powershell_parse.json'
$stream=[System.IO.File]::Open($destination,[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write,[System.IO.FileShare]::Read)
try {
    $bytes=[System.Text.Encoding]::UTF8.GetBytes(([ordered]@{passed=$true;files=$files;PSVersion=$PSVersionTable.PSVersion.ToString()} | ConvertTo-Json -Depth 5))
    $stream.Write($bytes,0,$bytes.Length)
} finally {$stream.Dispose()}
Write-Output 'Both actual launchers parse successfully.'
