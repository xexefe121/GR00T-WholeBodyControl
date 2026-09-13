param([Parameter(Mandatory=$true)][string]$Path)
$stream=[IO.File]::Open($Path,[IO.FileMode]::Open,[IO.FileAccess]::Read,([IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete))
try{$reader=New-Object IO.StreamReader($stream);try{$reader.ReadToEnd()}finally{$reader.Dispose()}}finally{$stream.Dispose()}
