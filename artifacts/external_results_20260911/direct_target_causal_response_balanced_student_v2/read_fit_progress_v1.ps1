$ErrorActionPreference='Stop'
$shareMode=[IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete
foreach($name in @('fit_process_v1/start.json','fit_process_v1/child.json','fit/progress.json','fit/failure.json','fit/setup_failure.json','fit/report.json','fit_process_v1/exit.json')){
    $path=Join-Path $PSScriptRoot $name
    if(Test-Path -LiteralPath $path){
        $stream=[IO.File]::Open($path,[IO.FileMode]::Open,[IO.FileAccess]::Read,$shareMode)
        try{$reader=[IO.StreamReader]::new($stream);try{$value=$reader.ReadToEnd()|ConvertFrom-Json;[ordered]@{path=$name;value=$value}|ConvertTo-Json -Depth 20 -Compress}finally{$reader.Dispose()}}finally{$stream.Dispose()}
    }
}
