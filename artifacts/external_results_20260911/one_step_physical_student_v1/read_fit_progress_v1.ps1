$ErrorActionPreference='Stop'
$runRoot=$PSScriptRoot
$shareMode=[IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete
function Read-AtomicJson([string]$Path){
    # Sharing DELETE permits the writer's atomic replacement while this handle exists.
    $stream=[IO.File]::Open($Path,[IO.FileMode]::Open,[IO.FileAccess]::Read,$shareMode)
    try {
        $reader=[IO.StreamReader]::new($stream,[Text.Encoding]::UTF8,$true)
        try { return ($reader.ReadToEnd() | ConvertFrom-Json) } finally { $reader.Dispose() }
    } finally { $stream.Dispose() }
}
$result=[ordered]@{observed_utc=[DateTime]::UtcNow.ToString('o');observation_only=$true;files=[ordered]@{}}
foreach($relative in @('fit_process/start.json','fit_process/child.json','fit/progress.json','fit/attempt_status.json','fit_process/exit.json')){
    $path=Join-Path $runRoot $relative
    if([IO.File]::Exists($path)){
        try { $result.files[$relative]=Read-AtomicJson $path }
        catch { $result.files[$relative]=[ordered]@{temporarily_unavailable=$true;error=$_.Exception.Message} }
    }
}
$result | ConvertTo-Json -Depth 100
