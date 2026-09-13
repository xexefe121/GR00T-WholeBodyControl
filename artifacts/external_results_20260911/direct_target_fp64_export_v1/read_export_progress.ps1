$ErrorActionPreference='Stop'
$shareMode=[IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete
foreach($name in @('export_process/start.json','export_process/child.json','export/progress.json','export/failure.json','export_process/exit.json')){
    $path=Join-Path $PSScriptRoot $name
    if(Test-Path -LiteralPath $path){
        $stream=[IO.File]::Open($path,[IO.FileMode]::Open,[IO.FileAccess]::Read,$shareMode)
        try{$reader=[IO.StreamReader]::new($stream);try{$value=$reader.ReadToEnd()|ConvertFrom-Json;[ordered]@{path=$name;value=$value}|ConvertTo-Json -Depth 20 -Compress}finally{$reader.Dispose()}}finally{$stream.Dispose()}
    }
}
