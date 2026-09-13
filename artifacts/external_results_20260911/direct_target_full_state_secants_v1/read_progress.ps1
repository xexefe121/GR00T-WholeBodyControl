$path='E:\codex-artifacts\sonic23_teleop_resume_20260911\direct_target_full_state_secants_v1\generation\progress.json'
if([IO.File]::Exists($path)){
    $stream=[IO.File]::Open($path,[IO.FileMode]::Open,[IO.FileAccess]::Read,([IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete))
    $reader=[IO.StreamReader]::new($stream)
    try{$reader.ReadToEnd()}finally{$reader.Dispose();$stream.Dispose()}
}
