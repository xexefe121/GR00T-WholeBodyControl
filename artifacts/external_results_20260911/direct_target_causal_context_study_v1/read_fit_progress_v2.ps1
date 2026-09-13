$ErrorActionPreference='Stop'
$shareMode=[IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete
foreach($name in @('fit_process_v2/start.json','fit_process_v2/child.json','fit/blinded/progress.json','fit/causal/progress.json','fit/blinded/failure.json','fit/causal/failure.json','fit/paired_failure.json','fit/paired_report.json','fit_process_v2/exit.json')){
    $path=Join-Path $PSScriptRoot $name
    if(Test-Path -LiteralPath $path){
        $stream=[IO.File]::Open($path,[IO.FileMode]::Open,[IO.FileAccess]::Read,$shareMode)
        try{$reader=[IO.StreamReader]::new($stream);try{$value=$reader.ReadToEnd()|ConvertFrom-Json;[ordered]@{path=$name;value=$value}|ConvertTo-Json -Depth 20 -Compress}finally{$reader.Dispose()}}finally{$stream.Dispose()}
    }
}
